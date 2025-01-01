import torch
from typing import Union, Tuple
import numpy as np

# local imports
from learned_simulator import time_diff


def get_random_walk_noise_for_position_sequence(
    position_sequence: torch.Tensor,
    noise_std_last_step: float
) -> torch.Tensor:
    """Generate random walk noise to apply to a position sequence.

    This function generates noise that follows a random walk pattern in velocity
    space, which is then integrated to get position noise. The noise standard
    deviation at the last step is controlled by noise_std_last_step.

    Args:
        position_sequence: Tensor of particle positions over time
            Shape: [num_particles, num_timesteps, num_dimensions]
        noise_std_last_step: Target standard deviation of noise at the last step

    Returns:
        Tensor of position noise with same shape as position_sequence
            Shape: [num_particles, num_timesteps, num_dimensions]

    Note:
        The noise is generated as a random walk in velocity space and then
        integrated to get position noise. The noise at each step is scaled
        so that the final step has the desired standard deviation.
    """
    # calculate velocity sequence from positions
    velocity_sequence = time_diff(position_sequence)

    # calculate number of velocity steps
    num_velocities = velocity_sequence.shape[1]

    # generate velocity noise
    # scale is set so that accumulated noise at last step has desired std
    step_noise_std = noise_std_last_step / np.sqrt(num_velocities)
    velocity_noise = torch.randn_like(
        velocity_sequence,
        dtype=position_sequence.dtype,
        device=position_sequence.device
    ) * step_noise_std

    # accumulate velocity noise over time
    velocity_noise.cumsum_(dim=1)

    # integrate velocity noise to get position noise
    # start with zero noise at first position
    position_noise = torch.zeros_like(position_sequence)
    position_noise[:, 1:] = velocity_noise.cumsum(dim=1)

    return position_noise


def validate_noise_inputs(
    position_sequence: torch.Tensor,
    noise_std_last_step: float
) -> None:
    """Validate inputs to noise generation function.

    Args:
        position_sequence: Position sequence tensor
        noise_std_last_step: Noise standard deviation

    Raises:
        ValueError: If inputs are invalid
    """
    if not isinstance(position_sequence, torch.Tensor):
        raise ValueError("position_sequence must be a torch.Tensor")
    
    if position_sequence.dim() != 3:
        raise ValueError(
            f"position_sequence must have 3 dimensions, got {position_sequence.dim()}"
        )
    
    if noise_std_last_step < 0:
        raise ValueError(
            f"noise_std_last_step must be non-negative, got {noise_std_last_step}"
        )
