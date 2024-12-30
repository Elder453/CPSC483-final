import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class GCN(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        dropout=0.5,
        save_mem=True,
        use_bn=True,
        use_residual=False
    ):
        super(GCN, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=not save_mem))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=not save_mem))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=not save_mem))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_residual = use_residual

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x = data.graph['node_feat']

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        x = self.convs[0](x, data.graph['edge_index'])
        x = self.activation(x)
        if self.use_bn:
            x = self.bns[0](x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i, conv in enumerate(self.convs[1:-1]):
            x_ = conv(x, data.graph['edge_index'])
            if self.use_bn:
                x_ = self.bns[i](x_)
            x_ = self.activation(x_)
            x_ = F.dropout(x_, p=self.dropout, training=self.training)
            if self.use_residual:
                x = x_ + x
            else:
                x = x_
        
        x = self.convs[-1](x, data.graph['edge_index'])
        return x


class GAT(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        dropout=0.5,
        use_bn=False,
        heads=2,
        out_heads=1
    ):
        super(GAT, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(
                in_channels, 
                hidden_channels, 
                dropout=dropout, 
                heads=heads, 
                concat=True,
                add_self_loops=True,
                negative_slope=0.2
            )
        )

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*heads))
        for _ in range(num_layers - 2):
            self.convs.append(
                    GATConv(
                        hidden_channels*heads,
                        hidden_channels,
                        dropout=dropout,
                        heads=heads,
                        concat=True,
                        add_self_loops=True,
                        negative_slope=0.2
                    )
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels*heads))

        self.convs.append(
            GATConv(
                hidden_channels*heads,
                out_channels,
                dropout=dropout,
                heads=out_heads,
                concat=False,
                add_self_loops=True,
                negative_slope=0.2
            )
        )

        self.dropout = dropout
        self.activation = F.elu 
        self.use_bn = use_bn
    
    def forward(self, data):
        x = data.graph['node_feat']
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, data.graph['edge_index'])
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, data.graph['edge_index'])
        return x

    def reset_parameters(self):
        for conv in self.convs:
            # Xavier initialization for better stability
            nn.init.xavier_uniform_(conv.lin.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
        for bn in self.bns:
            bn.reset_parameters()
