import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, NamedTuple
import numpy as np

# local imports
from utils_connectivity import compute_connectivity_for_batch
from dataloader import NCDataset
from graph_network import EncodeProcessDecode
from utils import INPUT_SEQUENCE_LENGTH, STD_EPSILON


class NormalizationStats(NamedTuple):
    mean: np.ndarray
    std: np.ndarray


@torch.jit.script
def time_diff(input_sequence: torch.Tensor) -> torch.Tensor:
    """Compute time differences between consecutive positions."""
    return input_sequence[:, 1:] - input_sequence[:, :-1]


class LearnedSimulator(nn.Module):
    """Neural network-based physics simulator."""
    
    def __init__(
        self,
        num_dimensions: int,
        connectivity_radius: float,
        graph_network_kwargs: Dict,
        boundaries: List[Tuple[float, float]],
        normalization_stats: Dict[str, NormalizationStats],
        num_particle_types: int,
        device: str,
        particle_type_embedding_size: int,
        args,
    ):
        """Initialize the simulator.
        
        Args:
            num_dimensions: Number of spatial dimensions
            connectivity_radius: Radius for particle connectivity
            graph_network_kwargs: Parameters for the GNN
            boundaries: List of (min, max) bounds per dimension
            normalization_stats: Statistics for normalizing physical quantities
            num_particle_types: Number of particle types
            device: Target computation device
            particle_type_embedding_size: Size of particle type embeddings
            args: Additional configuration arguments
        """
        super().__init__()
        
        self._connectivity_radius = connectivity_radius
        self._num_particle_types = num_particle_types
        self._boundaries = boundaries
        self._normalization_stats = normalization_stats
        self._node_input_size = (INPUT_SEQUENCE_LENGTH + 1) * num_dimensions
        self._edge_input_size = num_dimensions + 1

        # init particle type embeddings if multiple types exist
        if self._num_particle_types > 1:
            self._particle_type_embedding = nn.Parameter(
                torch.FloatTensor(self._num_particle_types, particle_type_embedding_size),
                requires_grad=True
            ).to(device)
            self._node_input_size += particle_type_embedding_size

        # init GNN
        self._graph_network = EncodeProcessDecode(
            node_input_size=self._node_input_size,
            edge_input_size=self._edge_input_size,
            output_size=num_dimensions,
            device=device,
            args=args,
            **graph_network_kwargs
        ).to(device)

    def _encoder_preprocessor(
        self,
        position_sequence: torch.Tensor,
        n_node: torch.Tensor,
        global_context: Optional[torch.Tensor] = None,
        particle_types: Optional[torch.Tensor] = None
    ) -> NCDataset:
        """Prepare input data for the graph network.
        
        Args:
            position_sequence: Particle positions over time [num_particles, num_steps, dim]
            n_node: Number of particles per example
            global_context: Optional global features
            particle_types: Optional particle type indices
        """
        # get most recent positions and compute velocities
        most_recent_position = position_sequence[:, -1]
        velocity_sequence = time_diff(position_sequence)

        # compute connectivity graph
        senders, receivers, n_edge = compute_connectivity_for_batch(
            most_recent_position.detach().cpu().numpy(),
            n_node.cpu().numpy(),
            self._connectivity_radius,
            velocity_sequence.device
        )

        node_features = []

        # normalize velocities
        velocity_stats = self._normalization_stats['velocity']
        velocity_mean = torch.tensor(velocity_stats.mean, device=velocity_sequence.device)
        velocity_std = torch.tensor(velocity_stats.std, device=velocity_sequence.device)
        normalized_velocity_sequence = (velocity_sequence - velocity_mean) / velocity_std
        node_features.append(normalized_velocity_sequence.flatten(1, 2))

        # add boundary distances
        boundaries = torch.tensor(self._boundaries, dtype=torch.float32, device=most_recent_position.device)
        distance_to_lower = most_recent_position - torch.unsqueeze(boundaries[:, 0], 0)
        distance_to_upper = torch.unsqueeze(boundaries[:, 1], 0) - most_recent_position
        distance_to_boundaries = torch.cat([distance_to_lower, distance_to_upper], dim=1)
        normalized_distances = torch.clip(
            distance_to_boundaries / self._connectivity_radius,
            -1.0,
            1.0
        )
        node_features.append(normalized_distances)

        # add particle type embeddings if available
        if self._num_particle_types > 1 and particle_types is not None:
            particle_types = particle_types.to(self._particle_type_embedding.device)
            particle_type_embeddings = self._particle_type_embedding[particle_types]
            node_features.append(particle_type_embeddings.to(most_recent_position.device))

        edge_features = []
        
        # compute relative displacements and distances
        normalized_relative_displacements = (
            most_recent_position[senders] - most_recent_position[receivers]
        ) / self._connectivity_radius
        edge_features.append(normalized_relative_displacements)

        normalized_relative_distances = torch.norm(
            normalized_relative_displacements,
            dim=-1,
            keepdim=True
        )
        edge_features.append(normalized_relative_distances)

        # normalize global context if provided
        if global_context is not None:
            context_stats = self._normalization_stats["context"]
            global_context = (
                global_context - context_stats.mean
            ) / max(context_stats.std, STD_EPSILON)

        # create graph tuple
        graph_tuple = NCDataset("input_graphs")
        graph_tuple.graph = {
            'node_feat': torch.cat(node_features, dim=-1),
            'edge_feat': torch.cat(edge_features, dim=-1),
            'global': global_context,
            'n_node': n_node,
            'n_edge': n_edge,
            'edge_index': torch.stack([senders, receivers])
        }

        return graph_tuple

    def _decoder_postprocessor(
        self,
        normalized_acceleration: torch.Tensor,
        position_sequence: torch.Tensor
    ) -> torch.Tensor:
        """Convert normalized accelerations to positions using Euler integration."""
        # de-normalize accel
        acceleration_stats = self._normalization_stats["acceleration"]
        acceleration_mean = torch.tensor(acceleration_stats.mean, device=normalized_acceleration.device)
        acceleration_std = torch.tensor(acceleration_stats.std, device=normalized_acceleration.device)
        acceleration = (normalized_acceleration * acceleration_std) + acceleration_mean

        # Euler integration
        most_recent_position = position_sequence[:, -1]
        most_recent_velocity = most_recent_position - position_sequence[:, -2]
        
        new_velocity = most_recent_velocity + acceleration  # dt = 1
        new_position = most_recent_position + new_velocity  # dt = 1
        
        return new_position

    def _inverse_decoder_postprocessor(
        self,
        next_position: torch.Tensor,
        position_sequence: torch.Tensor
    ) -> torch.Tensor:
        """Convert positions to normalized accelerations."""
        previous_position = position_sequence[:, -1]
        previous_velocity = previous_position - position_sequence[:, -2]
        next_velocity = next_position - previous_position
        acceleration = next_velocity - previous_velocity

        # normalize accel
        acceleration_stats = self._normalization_stats['acceleration']
        acceleration_mean = torch.tensor(acceleration_stats.mean, device=acceleration.device)
        acceleration_std = torch.tensor(acceleration_stats.std, device=acceleration.device)
        return (acceleration - acceleration_mean) / acceleration_std

    def forward(
        self,
        position_sequence: torch.Tensor,
        n_particles_per_example: torch.Tensor,
        global_context: Optional[torch.Tensor] = None,
        particle_types: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass: predict next position given sequence of previous positions."""
        input_graphs_tuple = self._encoder_preprocessor(
            position_sequence,
            n_particles_per_example,
            global_context,
            particle_types
        )
        normalized_acceleration = self._graph_network(input_graphs_tuple)
        return self._decoder_postprocessor(normalized_acceleration, position_sequence)

    def get_predicted_and_target_normalized_accelerations(
        self,
        next_position: torch.Tensor,
        position_sequence_noise: torch.Tensor,
        position_sequence: torch.Tensor,
        n_particles_per_example: torch.Tensor,
        global_context: Optional[torch.Tensor] = None,
        particle_types: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get normalized accelerations for both predicted and target positions."""
        # add noise to pos sequence
        noisy_position_sequence = position_sequence + position_sequence_noise
        
        # get pred accels
        input_graphs_tuple = self._encoder_preprocessor(
            noisy_position_sequence,
            n_particles_per_example,
            global_context,
            particle_types
        )
        predicted_normalized_acceleration = self._graph_network(input_graphs_tuple)
        
        # get target accels
        next_position_adjusted = next_position + position_sequence_noise[:, -1]
        target_normalized_acceleration = self._inverse_decoder_postprocessor(
            next_position_adjusted,
            noisy_position_sequence
        )
        
        return predicted_normalized_acceleration, target_normalized_acceleration
