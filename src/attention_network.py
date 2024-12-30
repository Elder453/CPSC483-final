import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from typing import Optional, Tuple, Union

class GATConv(MessagePassing):
    """Graph Attention Layer with multi-head attention."""
    
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        **kwargs
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        # Linear transformations for multi-head attention
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize learnable parameters."""
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(
        self, 
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        # Linear transformation for all nodes
        x = self.lin(x).view(-1, self.heads, self.out_channels)

        # Propagate messages along edges
        out = self.propagate(edge_index, x=x, size=None)

        # Optionally concatenate or mean the attention heads
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if return_attention_weights:
            # TODO: Implement attention weight return
            pass

        return out

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, index: torch.Tensor, size_i: Optional[int]) -> torch.Tensor:
        """Constructs messages to send along edges."""
        # Compute attention coefficients
        x = torch.cat([x_i, x_j], dim=-1)  # Shape: [E, heads, 2 * out_channels]
        alpha = (x * self.att).sum(dim=-1)  # Shape: [E, heads]
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, size_i)

        # Apply dropout to attention coefficients
        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout)

        return x_j * alpha.unsqueeze(-1)

class MultiHeadGATLayer(nn.Module):
    """Multi-head GAT layer with residual connection and normalization."""
    
    def __init__(
        self,
        in_channels: int,
        heads: int,
        head_dim: int,
        dropout: float = 0.6,
        use_residual: bool = True
    ):
        super().__init__()
        
        self.gat = GATConv(
            in_channels=in_channels,
            out_channels=head_dim,
            heads=heads,
            concat=True,
            dropout=dropout
        )
        
        self.norm = nn.LayerNorm(heads * head_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual
        
        if use_residual and in_channels != heads * head_dim:
            self.residual_proj = nn.Linear(in_channels, heads * head_dim)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Store input for residual
        identity = self.residual_proj(x)
        
        # Apply GAT layer
        out = self.gat(x, edge_index)
        out = self.norm(out)
        out = F.elu(out)
        out = self.dropout(out)
        
        # Add residual if enabled
        if self.use_residual:
            out = out + identity
            
        return out

class SingleHeadGATLayer(nn.Module):
    """Single-head GAT layer with larger dimension."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        dropout: float = 0.6,
        use_residual: bool = True
    ):
        super().__init__()
        
        self.gat = GATConv(
            in_channels=in_channels,
            out_channels=hidden_dim,
            heads=1,
            concat=False,
            dropout=dropout
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual
        
        if use_residual and in_channels != hidden_dim:
            self.residual_proj = nn.Linear(in_channels, hidden_dim)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Store input for residual
        identity = self.residual_proj(x)
        
        # Apply GAT layer
        out = self.gat(x, edge_index)
        out = self.norm(out)
        out = F.elu(out)
        out = self.dropout(out)
        
        # Add residual if enabled
        if self.use_residual:
            out = out + identity
            
        return out

class EnhancedGAT(nn.Module):
    """Enhanced GAT model with alternating multi-head and single-head layers."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        out_channels: int = 4,  # [c, p, u, v] output
        dropout: float = 0.6
    ):
        super().__init__()
        
        # Paper architecture:
        # layers 1,3,5,7: 4-head attn with dim 4
        # layers 2,4,6: single-head attn with dim 16
        # layer 8: single-head output layer with dim out_channels
        
        self.layers = nn.ModuleList([
            # layer 1: multi-head (4 heads, dim 4)
            MultiHeadGATLayer(in_channels, heads=4, head_dim=4, dropout=dropout),
            
            # layer 2: single-head (dim 16)
            SingleHeadGATLayer(16, hidden_dim=16, dropout=dropout),
            
            # layer 3: multi-head
            MultiHeadGATLayer(16, heads=4, head_dim=4, dropout=dropout),
            
            # layer 4: single-head
            SingleHeadGATLayer(16, hidden_dim=16, dropout=dropout),
            
            # layer 5: multi-head
            MultiHeadGATLayer(16, heads=4, head_dim=4, dropout=dropout),
            
            # layer 6: single-head
            SingleHeadGATLayer(16, hidden_dim=16, dropout=dropout),
            
            # layer 7: multi-head
            MultiHeadGATLayer(16, heads=4, head_dim=4, dropout=dropout),
            
            # layer 8: output layer
            SingleHeadGATLayer(16, hidden_dim=out_channels, dropout=dropout, use_residual=False)
        ])
        
        # init residual connection mappings
        self.residual_maps = nn.ModuleList([
            nn.Linear(16, 16) for _ in range(3)  # 1 for each pair of layers
        ])
        
        # gradient clipping value
        self.clip_value = 1.0

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # apply gradient clipping to input
        x = torch.clamp(x, -self.clip_value, self.clip_value)
        
        residuals = []
        
        # forward pass through all layers
        for i, layer in enumerate(self.layers):
            # store residual every 2 layers
            if i % 2 == 0 and i < len(self.layers) - 2:
                residuals.append(self.residual_maps[i//2](x))
            
            # layer forward pass
            x = layer(x, edge_index)
            
            # add residual connection
            if i > 0 and i % 2 == 0 and i < len(self.layers) - 1:
                x = x + residuals[i//2 - 1]
            
            # apply gradient clipping after each layer
            x = torch.clamp(x, -self.clip_value, self.clip_value)
        
        return x
