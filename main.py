# main.py

# Standard library imports
import argparse
import math
import os
import pickle
import time
from collections import deque
from typing import Optional, Dict, Union, List

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

# Local imports
from learned_simulator import LearnedSimulator
from noise_utils import get_random_walk_noise_for_position_sequence
from dataloader import OneStepDataset, RolloutDataset, one_step_collate
from rollout import rollout
from utils import (
    fix_seed,
    _combine_std,
    _read_metadata,
    get_kinematic_mask,
    print_args,
    Stats,
    NUM_PARTICLE_TYPES,
    INPUT_SEQUENCE_LENGTH,
    device
)

def _get_simulator(
    model_kwargs: dict,
    metadata: dict,
    acc_noise_std: float,
    vel_noise_std: float,
    args
) -> 'LearnedSimulator':
    """Initialize simulator with proper normalization statistics.
    
    Args:
        model_kwargs: Dictionary containing model parameters like latent_size, mlp_hidden_size etc.
        metadata: Dictionary containing simulation metadata including dimensions, bounds etc.
        acc_noise_std: Standard deviation of acceleration noise
        vel_noise_std: Standard deviation of velocity noise
        args: Additional arguments passed from command line
        
    Returns:
        LearnedSimulator: Initialized simulator instance with proper normalization
    """
    cast = lambda v: np.array(v, dtype=np.float32)
    
    acceleration_stats = Stats(
        cast(metadata['acc_mean']),
        _combine_std(cast(metadata['acc_std']), acc_noise_std))
    
    velocity_stats = Stats(
        cast(metadata['vel_mean']),
        _combine_std(cast(metadata['vel_std']), vel_noise_std))
    
    normalization_stats = {'acceleration': acceleration_stats,
                          'velocity': velocity_stats}
    
    if 'context_mean' in metadata:
        context_stats = Stats(
            cast(metadata['context_mean']), cast(metadata['context_std']))
        normalization_stats['context'] = context_stats

    simulator = LearnedSimulator(
        num_dimensions=metadata['dim'],
        connectivity_radius=metadata['default_connectivity_radius'],
        graph_network_kwargs=model_kwargs,
        boundaries=metadata['bounds'],
        num_particle_types=NUM_PARTICLE_TYPES,
        normalization_stats=normalization_stats,
        device=device,
        particle_type_embedding_size=16,
        args=args,
    )
    return simulator

def eval_one_step(args: argparse.Namespace) -> None:
    """Evaluate model on single-step predictions.
    
    Loads a trained model and evaluates its performance on single-step predictions
    using the specified dataset split. Calculates MSE loss for both position and
    acceleration predictions.
    
    Args:
        args: Namespace containing:
            - dataset: Name of the dataset to evaluate on
            - eval_split: Which data split to use (train/valid/test)
            - batch_size: Batch size for evaluation
            - model_path: Path to saved model checkpoints
            - gnn_type: Type of GNN used
            - noise_std: Standard deviation for noise
            - message_passing_steps: Number of message passing steps
            
    Raises:
        ValueError: If no model checkpoint is found
    """
    # Data setup
    sequence_dataset = OneStepDataset(args.dataset, args.eval_split)
    sequence_dataloader = DataLoader(
        sequence_dataset, 
        collate_fn=one_step_collate, 
        batch_size=args.batch_size, 
        shuffle=False
    )

    # Model initialization
    metadata = _read_metadata(data_path=f"datasets/{args.dataset}")
    model_kwargs = dict(
        latent_size=128,
        mlp_hidden_size=128,
        mlp_num_hidden_layers=2,
        num_message_passing_steps=args.message_passing_steps,
    )
    
    # Initialize simulator
    simulator = _get_simulator(
        model_kwargs=model_kwargs,
        metadata=metadata,
        vel_noise_std=args.noise_std,
        acc_noise_std=args.noise_std,
        args=args
    )
    
    # Load model checkpoint
    checkpoint_path = f'{args.model_path}/{args.dataset}/{args.gnn_type}'
    checkpoint_file = None
    for file in os.listdir(checkpoint_path):
        if file.startswith('best_val_mse'):
            checkpoint_file = os.path.join(checkpoint_path, file)
            break
    
    if not checkpoint_file:
        raise ValueError("No checkpoint exists!")
    print(f"Load checkpoint from: {checkpoint_file}")
    
    simulator_state_dict = torch.load(checkpoint_file, map_location=device)
    simulator.load_state_dict(simulator_state_dict)

    # Evaluation loop
    mse_loss = F.mse_loss
    total_loss = []
    time_step = 0

    print("################### Begin Evaluate One Step #######################")
    with torch.no_grad():
        for features, labels in sequence_dataloader:
            # Move data to device
            labels = labels.to(device)
            target_next_position = labels
            
            # Move features to device
            features['positions'] = features['positions'].to(device)
            features['particle_types'] = features['particle_types'].to(device)
            features['n_particles_per_example'] = features['n_particles_per_example'].to(device)
            if 'step_context' in features:
                features['step_context'] = features['step_context'].to(device)

            # Generate noise and get predictions
            sampled_noise = get_random_walk_noise_for_position_sequence(
                features['positions'],
                noise_std_last_step=args.noise_std
            ).to(device)

            predicted_next_position = simulator(
                position_sequence=features['positions'],
                n_particles_per_example=features['n_particles_per_example'],
                particle_types=features['particle_types'],
                global_context=features.get('step_context')
            )

            pred_target = simulator.get_predicted_and_target_normalized_accelerations(
                next_position=target_next_position,
                position_sequence=features['positions'],
                position_sequence_noise=sampled_noise,
                n_particles_per_example=features['n_particles_per_example'],
                particle_types=features['particle_types'],
                global_context=features.get('step_context')
            )
            pred_acceleration, target_acceleration = pred_target

            # Calculate losses
            loss_mse = mse_loss(pred_acceleration, target_acceleration)
            one_step_position_mse = mse_loss(predicted_next_position, target_next_position)
            total_loss.append(one_step_position_mse)
            
            print(f"step: {time_step}\t loss_mse: {loss_mse:.2f}\t "
                  f"one_step_position_mse: {one_step_position_mse * 1e9:.2f}e-9.")
            time_step += 1

        # Calculate and print average loss
        average_loss = torch.tensor(total_loss).mean().item()
        print(f"Average one step loss is: {average_loss * 1e9}e-9.")

def eval_rollout(args):
    """Evaluate model on trajectory rollouts."""
    # data setup
    sequence_dataset = RolloutDataset(args.dataset, args.eval_split)
    sequence_dataloader = DataLoader(
        sequence_dataset,
        collate_fn=one_step_collate,
        batch_size=1,
        shuffle=False
    )

    metadata = _read_metadata(data_path=f"datasets/{args.dataset}")
    model_kwargs = dict(
        latent_size=128,
        mlp_hidden_size=128,
        mlp_num_hidden_layers=2,
        num_message_passing_steps=args.message_passing_steps,
    )

    simulator = _get_simulator(
        model_kwargs,
        metadata,
        vel_noise_std=args.noise_std,
        acc_noise_std=args.noise_std,
        args=args
    )

    num_steps = metadata['sequence_length'] - INPUT_SEQUENCE_LENGTH

    # load model checkpoint
    model_path = f'{args.model_path}/{args.dataset}/{args.gnn_type}'
    output_path = f'{args.output_path}/{args.dataset}/{args.gnn_type}'
    files = os.listdir(model_path)
    file_name = None

    for file in files:
        if file.startswith('best_val_mse'):
            file_name = os.path.join(model_path, file)
            break

    if not file_name:
        raise ValueError("No checkpoint exists!")
    else:
        print(f"Load checkpoint from: {file_name}")

    simulator_state_dict = torch.load(file_name, map_location=device)
    simulator.load_state_dict(simulator_state_dict)

    # evaluation loop
    mse_loss = F.mse_loss
    total_loss = []
    time_step = 0

    print("################### Begin Evaluate Rollout #######################")
    with torch.no_grad():
        for feature, _ in sequence_dataloader:
            feature['positions'] = feature['positions'].to(device)
            feature['particle_types'] = feature['particle_types'].to(device)
            feature['n_particles_per_example'] = feature['n_particles_per_example'].to(device)
            if 'step_context' in feature:
                feature['step_context'] = feature['step_context'].to(device)

            # run rollout
            rollout_op = rollout(simulator, feature, num_steps)
            rollout_op['metadata'] = metadata
            
            # calculate losses
            loss_mse = mse_loss(
                rollout_op['predicted_rollout'], 
                rollout_op['ground_truth_rollout']
            )
            total_loss.append(loss_mse)
            print(f"step: {time_step}\t rollout_loss_mse: {loss_mse * 1e3:.2f}e-3.")

            # save rollout results
            file_name = f'rollout_{args.eval_split}_{time_step}.pkl'
            file_name = os.path.join(output_path, file_name)
            print(f"Saving rollout file {time_step}.")
            with open(file_name, 'wb') as file:
                pickle.dump(rollout_op, file)
            time_step += 1

        average_loss = torch.tensor(total_loss).mean().item()
        print(f"Average rollout loss is: {average_loss * 1e3:.2f}e-3.")

def get_elapsed_time(start_time):
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    return f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"

def train(args: argparse.Namespace) -> None:
    """Train the simulator model.
    
    Performs training of the particle simulator using the provided configuration.
    Includes periodic validation and model checkpointing of best performing models.
    
    Args:
        args: Configuration namespace containing:
            - seed: Random seed for reproducibility
            - dataset: Name of dataset to train on
            - batch_size: Training batch size
            - noise_std: Standard deviation for noise injection
            - lr: Initial learning rate
            - weight_decay: Weight decay for optimizer
            - num_steps: Number of training steps
            - test_step: Steps between validation
            - model_path: Path to save model checkpoints
            - message_passing_steps: Number of message passing steps
            - gnn_type: Type of GNN model used
    """
    fix_seed(args.seed)

    # Data setup
    train_dataset = OneStepDataset(args.dataset, 'train')
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=one_step_collate,
        batch_size=args.batch_size,
        shuffle=True
    )

    # Add a small portion of the valid split to evaluate 
    # "one-step" MSE periodically as in the original paper.
    valid_dataset = OneStepDataset(args.dataset, 'valid')
    num_valid_samples = len(valid_dataset)
    eval_subset_size = min(5000, num_valid_samples) # Eval on ~15k samples

    # Calculate steps and epochs
    steps_per_epoch = len(train_dataloader) # steps == batches
    total_steps = args.num_steps
    validation_frequency = 5000
    print(f"\nTraining for {args.num_steps} steps with {steps_per_epoch} steps per epoch")
    print(f"Validating every {validation_frequency} steps")

    # Model initialization
    metadata = _read_metadata(data_path=f"datasets/{args.dataset}")
    model_kwargs = dict(
        latent_size=128,
        mlp_hidden_size=128,
        mlp_num_hidden_layers=2,
        num_message_passing_steps=args.message_passing_steps,
    )

    simulator = _get_simulator(
        model_kwargs=model_kwargs,
        metadata=metadata,
        vel_noise_std=args.noise_std,
        acc_noise_std=args.noise_std,
        args=args
    )

    # Learning rate schedule setup (exponential decay)
    decay_steps = int(25e5)
    min_lr = 1e-6

    # Optimization setup
    optimizer = torch.optim.Adam(
        simulator.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    mse_loss = F.mse_loss
    best_val_loss = float("inf")
    global_step = 0

    # Loss tracking setup
    val_mse_history = deque()
    val_steps_history = deque()

    # Track validation improvement as in original GNS approach
    patience = 150000
    min_delta = 1e-6 # hyperparameter
    steps_without_improvement = 0

    # Create fixed checkpoint path
    checkpoint_dir = f'{args.model_path}/{args.dataset}/{args.gnn_type}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = f'{checkpoint_dir}/best_val_mse.pt'

    start_time = time.time()
    # Training loop
    print("\nStarting training...")
    try:
        while global_step < args.num_steps:
            for features, labels in train_dataloader:
                if global_step >= total_steps:
                    break
                
                # Move data to device
                labels = labels.to(device)
                target_next_position = labels
                
                for key in ['positions', 'particle_types', 'n_particles_per_example']:
                    features[key] = features[key].to(device)
                if 'step_context' in features:
                    features['step_context'] = features['step_context'].to(device)

                # Add noise with masking
                sampled_noise = get_random_walk_noise_for_position_sequence(
                    features['positions'],
                    noise_std_last_step=args.noise_std
                ).to(device)
                
                non_kinematic_mask = torch.logical_not(get_kinematic_mask(features['particle_types']))
                noise_mask = non_kinematic_mask.unsqueeze(1).unsqueeze(2)
                sampled_noise *= noise_mask

                simulator.train()
                optimizer.zero_grad()

                # Forward pass and loss calculation
                pred_target = simulator.get_predicted_and_target_normalized_accelerations(
                    next_position=target_next_position,
                    position_sequence=features['positions'],
                    position_sequence_noise=sampled_noise,
                    n_particles_per_example=features['n_particles_per_example'],
                    particle_types=features['particle_types'],
                    global_context=features.get('step_context')
                )
                pred_acceleration, target_acceleration = pred_target

                # Calculate loss on non-kinematic particles
                loss = (pred_acceleration[non_kinematic_mask] - target_acceleration[non_kinematic_mask]) ** 2
                num_non_kinematic = torch.sum(non_kinematic_mask.to(torch.float32))
                loss = torch.sum(loss) / torch.sum(num_non_kinematic)

                # Optimization step
                loss.backward()
                clip_grad_norm_(simulator.parameters(), 1.0)
                optimizer.step()

                # Update LR with exp decay
                current_lr = (args.lr - min_lr) * (0.1 ** (global_step / decay_steps)) + min_lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

                # For logging: compute position MSE
                simulator.eval()
                with torch.no_grad():
                    predicted_next_position = simulator(
                        position_sequence=features['positions'],
                        n_particles_per_example=features['n_particles_per_example'],
                        particle_types=features['particle_types'],
                        global_context=features.get('step_context')
                    )
                    batch_loss_mse = mse_loss(pred_acceleration, target_acceleration).item()
                    batch_pos_mse = mse_loss(predicted_next_position, target_next_position).item()

                # Print progress every 250 steps
                if global_step % 250 == 0:
                    print(f"{get_elapsed_time(start_time)} "
                          f"Global Step: {global_step} | "
                          f"Train Position MSE: {batch_pos_mse:.4e} | "
                          f"Batch Accel MSE: {batch_loss_mse:.3f} | "
                          f"LR: {current_lr:.2e}")

                if global_step % validation_frequency == 0 and global_step > 0:
                    simulator.eval()
                    valid_indices = np.random.choice(num_valid_samples, eval_subset_size, replace=False)
                    valid_subset = Subset(valid_dataset, valid_indices)
                    val_dataloader = DataLoader(
                        valid_subset, 
                        collate_fn=one_step_collate,
                        batch_size=args.batch_size,
                        shuffle=False
                    )

                    val_losses = []
                    with torch.no_grad():
                        for vfeat, vlabels in val_dataloader:
                            vlabels = vlabels.to(device)
                            for key in ['positions', 'particle_types', 'n_particles_per_example']:
                                vfeat[key] = vfeat[key].to(device)
                            if 'step_context' in vfeat:
                                vfeat['step_context'] = vfeat['step_context'].to(device)

                            vpred = simulator(
                                position_sequence=vfeat['positions'],
                                n_particles_per_example=vfeat['n_particles_per_example'],
                                particle_types=vfeat['particle_types'],
                                global_context=vfeat.get('step_context')
                            )
                            val_losses.append(mse_loss(vpred, vlabels).item())

                    mean_val_loss = float(np.mean(val_losses))
                    val_mse_history.append(mean_val_loss)
                    val_steps_history.append(global_step)
                    print(f"{get_elapsed_time(start_time)} Validation One-step MSE: {mean_val_loss:.4e}")

                    # Check for improvement
                    if mean_val_loss < (best_val_loss - min_delta):
                        best_val_loss = mean_val_loss
                        steps_without_improvement = 0
                        print(f"New best validation MSE: {best_val_loss:.4e} -> Saving model...")
                        checkpoint = {
                            'model_state_dict': simulator.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'global_step': global_step,
                            'val_mse_history': list(val_mse_history),
                            'val_steps_history': list(val_steps_history),
                        }
                        torch.save(checkpoint, best_model_path)
                    else:
                        steps_without_improvement += validation_frequency
                        print(f"No improvement. Steps without improvement: {steps_without_improvement}/{patience}")

                    # Early stopping if patience is exceeded
                    if steps_without_improvement >= patience:
                        print("\nEarly stopping triggered - No improvement on validation set.")
                        return
                
                global_step += 1
    
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        print(f"Saving current checkpoint - Step: {global_step}")
        checkpoint = {
            'model_state_dict': simulator.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
            'val_mse_history': list(val_mse_history),
            'val_steps_history': list(val_steps_history),
        }
        torch.save(
            checkpoint,
            f'{checkpoint_dir}/interrupted_checkpoint.pt'
        )

    print(f"\n[{get_elapsed_time(start_time)}] Training completed!")
    print(f"Best loss achieved: {best_val_loss:.4e}")
    print(f"Best model saved at: {best_model_path}")

def parse_arguments():
    """Parse command line arguments organized by usage mode."""
    parser = argparse.ArgumentParser(description="Learning to Simulate.")
    
    # Global arguments (used by all modes)
    global_group = parser.add_argument_group('Global Arguments')
    global_group.add_argument('--mode', default='train',
                        choices=['train', 'eval', 'eval_rollout'],
                        help='Train model, one step evaluation or rollout evaluation.')
    global_group.add_argument('--dataset', default="Water", type=str,
                        help='The dataset directory.')
    global_group.add_argument('--batch_size', default=2, type=int,
                        help='The batch size for training and evaluation.')
    global_group.add_argument('--model_path', default="models", type=str,
                        help='The path for saving/loading model checkpoints.')
    global_group.add_argument('--gnn_type', default='gcn', 
                        choices=['gcn', 'gat', 'trans_gnn', 'interaction_net'],
                        help='The GNN to be used as processor.')
    global_group.add_argument('--message_passing_steps', default=10, type=int,
                        help='Number of GNN message passing steps.')
    global_group.add_argument('--noise_std', default=0.0003, type=float,
                        help='The std deviation of the noise for training and evaluation.')
    
    # Training-specific arguments
    train_group = parser.add_argument_group('Training Arguments (only used when mode=train)')
    train_group.add_argument('--num_steps', default=2e7, type=int,
                        help='Maximum number of training steps.')
    train_group.add_argument('--seed', type=int, default=483,
                        help='Random seed for reproducibility.')
    train_group.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate.')
    train_group.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay for optimizer.')
    
    # Evaluation-specific arguments
    eval_group = parser.add_argument_group('Evaluation Arguments (only used when mode=eval or eval_rollout)')
    eval_group.add_argument('--eval_split', default='test',
                        choices=['train', 'valid', 'test'],
                        help='Dataset split to use for evaluation.')
    eval_group.add_argument('--output_path', default="rollouts", type=str,
                        help='Path for saving rollout results (only used in eval_rollout mode).')
    
    # GNN Architecture arguments
    gnn_group = parser.add_argument_group('GNN Architecture Arguments')
    gnn_group.add_argument('--hidden_channels', type=int, default=32,
                        help='Number of hidden channels in GNN layers.')
    gnn_group.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate.')
    gnn_group.add_argument('--num_gnn_layers', type=int, default=2,
                        help='Number of GNN layers.')
    
    # GAT-specific arguments
    gat_group = parser.add_argument_group('GAT-specific Arguments (only used when gnn_type=gat)')
    gat_group.add_argument('--gat_heads', type=int, default=8,
                        help='Number of attention heads for GAT.')
    gat_group.add_argument('--out_heads', type=int, default=1,
                        help='Number of output heads for GAT.')
    
    # TransGNN-specific arguments
    trans_group = parser.add_argument_group('TransGNN-specific Arguments (only used when gnn_type=trans_gnn)')
    trans_group.add_argument('--use_bn', action='store_true',
                        help='Use layer normalization.')
    trans_group.add_argument('--dropedge', type=float, default=0.0,
                        help='Edge dropout rate for regularization.')
    trans_group.add_argument('--dropnode', type=float, default=0.0,
                        help='Node dropout rate for regularization.')
    trans_group.add_argument('--trans_heads', type=int, default=4,
                        help='Number of transformer heads.')
    trans_group.add_argument('--nb_random_features', type=int, default=30,
                        help='Number of random features.')
    trans_group.add_argument('--use_gumbel', action='store_true',
                        help='Use Gumbel softmax for message passing.')
    trans_group.add_argument('--use_residual', action='store_true',
                        help='Use residual connections for each GNN layer.')
    trans_group.add_argument('--nb_sample_gumbel', type=int, default=10,
                        help='Number of samples for Gumbel softmax sampling.')
    trans_group.add_argument('--temperature', type=float, default=0.25,
                        help='Temperature coefficient for softmax.')
    trans_group.add_argument('--reg_weight', type=float, default=0.1,
                        help='Weight for graph regularization.')
    trans_group.add_argument('--projection_matrix_type', type=bool, default=True,
                        help='Use projection matrix.')
    
    args = parser.parse_args()
    
    # Create output directories if they don't exist
    os.makedirs(f'{args.model_path}/{args.dataset}/{args.gnn_type}',
               exist_ok=True)
    if args.mode == 'eval_rollout':
        os.makedirs(f'{args.output_path}/{args.dataset}/{args.gnn_type}',
                   exist_ok=True)
    
    print_args(args)
    return args

if __name__ == '__main__':
    args = parse_arguments()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        eval_one_step(args)
    elif args.mode == 'eval_rollout':
        eval_rollout(args)
    else:
        raise ValueError("Unrecognized mode!")
