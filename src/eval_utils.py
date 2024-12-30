import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import wasserstein_distance

from utils import device


def compute_one_step_metrics(simulator, features, labels):
    """Compute one-step prediction metrics: acceleration MSE and position MSE.
    
    Returns:
        tuple: (accel_mse, pos_mse) containing:
            - accel_mse: MSE between predicted and target accelerations
            - pos_mse: MSE between predicted and target positions
    """
    # Get position prediction
    predicted_next_position = simulator(
        position_sequence=features['positions'],
        n_particles_per_example=features['n_particles_per_example'],
        particle_types=features['particle_types'],
        global_context=features.get('step_context')
    )

    # Get acceleration predictions and targets
    pred_target = simulator.get_predicted_and_target_normalized_accelerations(
        next_position=labels,  # target_next_position
        position_sequence=features['positions'],
        position_sequence_noise=torch.zeros_like(features['positions']), # do NOT add noise to eval
        n_particles_per_example=features['n_particles_per_example'],
        particle_types=features['particle_types'],
        global_context=features.get('step_context')
    )
    pred_acceleration, target_acceleration = pred_target

    # Calculate losses
    accel_mse = F.mse_loss(pred_acceleration, target_acceleration).item()
    pos_mse = F.mse_loss(predicted_next_position, labels).item()

    return pos_mse, accel_mse


def compute_mse_n(simulator, features, rollout_op, n_frames=20):
    """Compute MSE averaged across n-frame intervals in rollout.
    Computes both position and acceleration MSE.
    
    Args:
        simulator: Neural network physics simulator
        features: Input features dictionary
        rollout_op: Dictionary containing rollout results
        n_frames: Number of frames to sample (default 20)
        
    Returns:
        tuple: (pos_mse, accel_mse) for sampled frames
    """
    pred_rollout = rollout_op['predicted_rollout']
    gt_rollout = rollout_op['ground_truth_rollout']
    pred_accel = rollout_op['predicted_accelerations']
    gt_accel = rollout_op['target_accelerations']
    
    # Sample every n frames
    indices = torch.arange(0, pred_rollout.size(0), n_frames)
    pred_pos_samples = pred_rollout[indices]
    gt_pos_samples = gt_rollout[indices]
    pred_accel_samples = pred_accel[indices]
    gt_accel_samples = gt_accel[indices]
    
    # Compute MSE averaged across time, particles and spatial dimensions
    pos_mse = F.mse_loss(pred_pos_samples, gt_pos_samples).item()
    accel_mse = F.mse_loss(pred_accel_samples, gt_accel_samples).item()
    
    return pos_mse, accel_mse


def compute_mse_full(simulator, features, rollout_op):
    """Compute MSE averaged across full rollout.
    Computes both position and acceleration MSE.
    
    Returns:
        tuple: (pos_mse, accel_mse) for full rollout
    """
    pred_rollout = rollout_op['predicted_rollout']
    gt_rollout = rollout_op['ground_truth_rollout']
    pred_accel = rollout_op['predicted_accelerations']
    gt_accel = rollout_op['target_accelerations']
    
    pos_mse = F.mse_loss(pred_rollout, gt_rollout).item()
    accel_mse = F.mse_loss(pred_accel, gt_accel).item()
    
    return pos_mse, accel_mse


def compute_emd(simulator, features, rollout_op):
    """Compute Earth Mover's Distance between predicted and ground truth distributions.
    Uses scipy's wasserstein_distance implementation.
    """
    
    pred_rollout = rollout_op['predicted_rollout'].cpu().numpy()
    gt_rollout = rollout_op['ground_truth_rollout'].cpu().numpy()
    
    # Initialize total EMD
    total_emd = 0.0
    n_timesteps = pred_rollout.shape[0]
    
    for t in range(n_timesteps):
        # Get positions at current timestep
        pred_pos = pred_rollout[t]  # [num_particles, 2]
        gt_pos = gt_rollout[t]      # [num_particles, 2]
        
        # Compute EMD separately for x and y coordinates
        emd_x = wasserstein_distance(pred_pos[:, 0], gt_pos[:, 0])
        emd_y = wasserstein_distance(pred_pos[:, 1], gt_pos[:, 1])
        
        # Take average of x and y EMDs
        timestep_emd = (emd_x + emd_y) / 2
        total_emd += timestep_emd
    
    # Return average EMD across timesteps
    return total_emd / n_timesteps
