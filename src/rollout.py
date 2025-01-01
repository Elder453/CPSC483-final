import torch
from typing import Dict, List, Optional, Tuple, Union
from torch import Tensor

# local imports
from utils import INPUT_SEQUENCE_LENGTH, get_kinematic_mask


def rollout(
    simulator: torch.nn.Module,
    features: Dict[str, Union[Tensor, List[Tensor]]],
    num_steps: int
) -> Dict[str, Tensor]:
    """Simulates a physics trajectory by applying the model sequentially.
    
    Args:
        simulator: Neural network physics simulator
        features: Dictionary containing:
            - positions: Particle positions [num_particles, num_timesteps, dims]
            - particle_types: Particle type IDs
            - n_particles_per_example: Particles per simulation
            - step_context: Optional global features per timestep
        num_steps: Number of simulation steps to run
    
    Returns:
        Dictionary containing:
            - initial_positions: Starting positions [sequence_length, num_particles, dims]
            - predicted_rollout: Predicted trajectory [num_steps, num_particles, dims]
            - ground_truth_rollout: Actual trajectory [num_steps, num_particles, dims]
            - particle_types: Particle type IDs
            - global_context: Optional global features
    """
    # extract initial conditions
    initial_positions = features['positions'][:, 0:INPUT_SEQUENCE_LENGTH]
    ground_truth_positions = features['positions'][:, INPUT_SEQUENCE_LENGTH:]
    global_context = features.get('step_context')

    # store acceleration
    predicted_accelerations = []
    target_accelerations = []

    def step_fn(
        step: int,
        current_positions: Tensor,
        predictions: List[Tensor]
    ) -> Tuple[int, Tensor, List[Tensor]]:
        """Performs one step of the simulation.
        
        Args:
            step: Current simulation step
            current_positions: Current particle positions
            predictions: List of predicted positions
            
        Returns:
            Tuple of (next step, next positions, updated predictions)
        """
        # get global context for current step if available
        if global_context is None:
            global_context_step = None
        else:
            global_context_step = global_context[step + INPUT_SEQUENCE_LENGTH - 1].unsqueeze(0)

        # predict next position
        next_position = simulator(
            current_positions,
            n_particles_per_example=features['n_particles_per_example'],
            particle_types=features['particle_types'],
            global_context=global_context_step
        )

        # compute accelerations for current step
        pred_target = simulator.get_predicted_and_target_normalized_accelerations(
            next_position=ground_truth_positions[:, step],  # target pos
            position_sequence=current_positions,
            position_sequence_noise=torch.zeros_like(current_positions),  # no noise during rollout
            n_particles_per_example=features['n_particles_per_example'],
            particle_types=features['particle_types'],
            global_context=global_context_step
        )
        pred_accel, target_accel = pred_target
        
        # store acceleration data
        predicted_accelerations.append(pred_accel)
        target_accelerations.append(target_accel)

        # apply kinematic constraints
        kinematic_mask = get_kinematic_mask(features['particle_types']).unsqueeze(1).tile(2)
        next_position_ground_truth = ground_truth_positions[:, step]
        next_position = torch.where(kinematic_mask, next_position_ground_truth, next_position)
        predictions.append(next_position)

        # update position sequence
        next_positions = torch.cat([
            current_positions[:, 1:],   # remove oldest position
            next_position.unsqueeze(1)  # add new position
        ], dim=1)

        return step + 1, next_positions, predictions

    # run simulation
    current_step = 0
    predictions: List[Tensor] = []
    current_positions = initial_positions

    for _ in range(num_steps):
        current_step, current_positions, predictions = step_fn(
            current_step, current_positions, predictions
        )

    # prepare output
    output_dict = {
        'initial_positions': torch.transpose(initial_positions, 0, 1),
        'predicted_rollout': torch.stack(predictions),
        'ground_truth_rollout': torch.transpose(ground_truth_positions, 0, 1),
        'particle_types': features['particle_types'],
        'predicted_accelerations': torch.stack(predicted_accelerations),
        'target_accelerations': torch.stack(target_accelerations),
    }

    if global_context is not None:
        output_dict['global_context'] = global_context

    return output_dict
