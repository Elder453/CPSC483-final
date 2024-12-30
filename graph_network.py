import torch
import torch.nn as nn
from torch_scatter import scatter
from dataloader import NCDataset
from models import GCN, GAT
from message_passing import EdgeModel, NodeModel, GraphNetwork
from typing import Optional, Dict, Union, List
from dataclasses import dataclass


class MLP(nn.Module):
    """Multi-layer perceptron with configurable hidden layers."""
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        output_size: int
    ):
        super().__init__()
        self.lins = nn.ModuleList()
        
        if num_hidden_layers == 0:
            self.lins.append(nn.Linear(input_size, output_size))
        else:
            self.lins.append(nn.Linear(input_size, hidden_size))
            for _ in range(num_hidden_layers - 1):
                self.lins.append(nn.Linear(hidden_size, hidden_size))
            self.lins.append(nn.Linear(hidden_size, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lin in self.lins:
            x = lin(x)
        return x


class EncodeProcessDecode(nn.Module):
    """Encode-Process-Decode architecture for graph neural networks."""
    
    def __init__(
        self,
        node_input_size: int,
        edge_input_size: int,
        latent_size: int,
        mlp_hidden_size: int,
        mlp_num_hidden_layers: int,
        num_message_passing_steps: int,
        output_size: int,
        device: str,
        args,
    ):
        super().__init__()
        
        # store config
        self._node_input_size = node_input_size
        self._edge_input_size = edge_input_size
        self._latent_size = latent_size
        self._mlp_hidden_size = mlp_hidden_size
        self._mlp_num_hidden_layers = mlp_num_hidden_layers
        self._num_message_passing_steps = num_message_passing_steps
        self._output_size = output_size
        self.device = device
        self.args = args
        
        # build network components
        self._network_builder()

    def _build_mlp_with_layer_norm(self, input_size: int) -> nn.Sequential:
        """Creates an MLP followed by layer normalization."""
        mlp = MLP(
            input_size=input_size,
            hidden_size=self._mlp_hidden_size,
            num_hidden_layers=self._mlp_num_hidden_layers,
            output_size=self._latent_size
        )
        return nn.Sequential(mlp, nn.LayerNorm(self._latent_size))

    def _network_builder(self):
        """Builds all network components."""
        # build encoders
        self.node_encoder = self._build_mlp_with_layer_norm(self._node_input_size)
        self.edge_encoder = self._build_mlp_with_layer_norm(self._edge_input_size)

        # build processor networks
        self._processor_networks = nn.ModuleList()
        for _ in range(self._num_message_passing_steps):
            if self.args.gnn_type == 'gcn':
                processor = GCN(
                    self._latent_size, 
                    self.args.hidden_channels,
                    self._latent_size, 
                    self.args.num_gnn_layers,
                    self.args.dropout, 
                    use_bn=self.args.use_bn
                ).to(self.device)
            elif self.args.gnn_type == 'gat':
                processor = GAT(
                    self._latent_size, 
                    self.args.hidden_channels,
                    self._latent_size, 
                    self.args.num_gnn_layers,
                    self.args.dropout, 
                    self.args.use_bn,
                    self.args.gat_heads, 
                    self.args.out_heads
                ).to(self.device)
            elif self.args.gnn_type == 'interaction_net':
                processor = GraphNetwork(
                    edge_model=EdgeModel(self._build_mlp_with_layer_norm(self._latent_size * 3)),
                    node_model=NodeModel(self._build_mlp_with_layer_norm(self._latent_size * 2)),
                ).to(self.device)
            else:
                raise ValueError(f"Unsupported GNN type: {self.args.gnn_type}")
            
            self._processor_networks.append(processor)

        # build decoder
        self._decoder_network = MLP(
            input_size=self._latent_size,
            hidden_size=self._mlp_hidden_size,
            num_hidden_layers=self._mlp_num_hidden_layers,
            output_size=self._output_size
        )

    def _encode(self, input_graph: NCDataset) -> NCDataset:
        """Encodes input graph into latent representation."""
        if input_graph.graph['global'] is not None:
            input_graph.graph['node_feat'] = torch.cat(
                [input_graph.graph['node_feat'], input_graph.graph['global']], 
                dim=-1
            )

        # create latent graph
        latent_graph_0 = NCDataset("latent_graph_0")
        latent_graph_0.graph = {
            'node_feat': self.node_encoder(input_graph.graph['node_feat']),
            'edge_feat': self.edge_encoder(input_graph.graph['edge_feat']),
            'global': None,
            'n_node': input_graph.graph['n_node'],
            'n_edge': input_graph.graph['n_edge'],
            'edge_index': input_graph.graph['edge_index'].to(input_graph.graph['n_node'].device),
        }

        # aggregate edge features to nodes
        num_nodes = torch.sum(latent_graph_0.graph['n_node']).item()
        latent_graph_0.graph['node_feat'] += scatter(
            latent_graph_0.graph['edge_feat'],
            latent_graph_0.graph['edge_index'][0],
            dim=0,
            dim_size=num_nodes,
            reduce='mean'
        )
        
        return latent_graph_0

    def _process_step(self, processor_network_k: nn.Module, 
                     latent_graph_prev_k: NCDataset) -> NCDataset:
        """Performs one step of message passing."""
        if self.args.gnn_type == 'interaction_net':
            new_node_feature, new_edge_feature = processor_network_k(latent_graph_prev_k.graph)
            latent_graph_k = NCDataset('latent_graph_k')
            latent_graph_k.graph = {
                'node_feat': latent_graph_prev_k.graph['node_feat'] + new_node_feature,
                'edge_feat': latent_graph_prev_k.graph['edge_feat'] + new_edge_feature,
                'global': None,
                'n_node': latent_graph_prev_k.graph['n_node'],
                'n_edge': latent_graph_prev_k.graph['n_edge'],
                'edge_index': latent_graph_prev_k.graph['edge_index']
            }
        else:
            new_node_feature = processor_network_k(latent_graph_prev_k)
            latent_graph_k = NCDataset('latent_graph_k')
            latent_graph_k.graph = {
                'node_feat': latent_graph_prev_k.graph['node_feat'] + new_node_feature,
                'edge_feat': None,
                'global': None,
                'n_node': latent_graph_prev_k.graph['n_node'],
                'n_edge': latent_graph_prev_k.graph['n_edge'],
                'edge_index': latent_graph_prev_k.graph['edge_index']
            }
        return latent_graph_k

    def _process(self, latent_graph_0: NCDataset) -> NCDataset:
        """Processes the latent graph through multiple message passing steps."""
        latent_graph_prev_k = latent_graph_0
        latent_graph_k = latent_graph_0
        
        for processor_network_k in self._processor_networks:
            latent_graph_k = self._process_step(processor_network_k, latent_graph_prev_k)
            latent_graph_prev_k = latent_graph_k

        return latent_graph_k

    def _decode(self, latent_graph: NCDataset) -> torch.Tensor:
        """Decodes latent graph to output."""
        return self._decoder_network(latent_graph.graph['node_feat'])

    def forward(self, input_graph: NCDataset) -> torch.Tensor:
        """Forward pass through the full encode-process-decode pipeline."""
        latent_graph_0 = self._encode(input_graph)
        latent_graph_m = self._process(latent_graph_0)
        return self._decode(latent_graph_m)
