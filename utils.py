# utils.py

# Standard library imports
import collections
import json
import os
import random
import time
from typing import Dict, Any, Union, NamedTuple, List

# Third-party imports
import numpy as np
import torch
from tabulate import tabulate

# Type definitions
Stats = collections.namedtuple('Stats', ['mean', 'std'])

# Constants
INPUT_SEQUENCE_LENGTH = 6  # For calculating last 5 velocities
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def print_args(args) -> None:
    """Print arguments in a formatted table."""
    args_dict = vars(args)
    keys = sorted(args_dict.keys())
    
    rows = [[k.replace("_", " ").capitalize(), args_dict[k]] for k in keys]
    
    print(tabulate(rows, headers=["Parameter", "Value"], tablefmt="grid"))

def fix_seed(seed: int) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_elapsed_time(start_time):
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    return f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"

def get_kinematic_mask(particle_types: torch.Tensor) -> torch.Tensor:
    """Get boolean mask for kinematic particles."""
    return torch.eq(particle_types, KINEMATIC_PARTICLE_ID)

def _combine_std(std_x: float, std_y: float) -> float:
    """Combine standard deviations using root sum of squares."""
    return np.sqrt(std_x**2 + std_y**2)

def compute_multi_step_loss(
    preds: List[torch.Tensor],
    targets: List[torch.Tensor],
    non_kinematic_mask: torch.Tensor
) -> torch.Tensor:
    """Compute multi-step loss across prediction steps.
    
    Computes loss only on non-kinematic particles and averages across:
    1. All particles for each step
    2. All steps in the sequence
    
    Args:
        preds: List of predicted accelerations or positions [steps]
            Each tensor has shape [num_particles, dims]
        targets: List of target accelerations or positions [steps]
            Each tensor has shape [num_particles, dims]
        non_kinematic_mask: Boolean mask identifying non-kinematic particles
            Shape [num_particles]
        
    Returns:
        Average loss across steps and non-kinematic particles
        
    Raises:
        ValueError: If prediction and target lists have different lengths
        ValueError: If any acceleration tensors have inconsistent shapes
    """
    # Validate inputs
    if len(preds) != len(targets):
        raise ValueError(
            f"Prediction and target lists must have same length."
            f"Got {len(preds)} and {len(targets)}"
        )
    
    # Initialize total loss
    total_loss = torch.tensor(0.0, device=device)
    
    # Accumulate loss for each step
    for pred, target in zip(preds, targets):
        # Validate tensor shapes
        if pred.shape != target.shape:
            raise ValueError(
                f"Prediction and target shapes must match. "
                f"Got {pred.shape} and {target.shape}"
            )
            
        # Compute squared error for non-kinematic particles
        step_loss = (pred[non_kinematic_mask] - target[non_kinematic_mask]) ** 2
        total_loss += torch.sum(step_loss)
    
    # Average loss across both steps and particles
    num_non_kinematic = torch.sum(non_kinematic_mask.to(torch.float32))
    avg_loss = total_loss / (num_non_kinematic * len(preds))
    
    return avg_loss

def _read_metadata(data_path: str) -> Dict[str, Any]:
    """Read metadata from JSON file."""
    metadata_path = os.path.join(data_path, 'metadata.json')
    with open(metadata_path, 'rt') as fp:
        return json.loads(fp.read())
