# standard library imports
import argparse
import os
import pickle
import signal
import time
from collections import deque
from typing import Dict, List

# 3rd-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset

# local imports
from learned_simulator import LearnedSimulator
from utils_noise import get_random_walk_noise_for_position_sequence
from dataloader import OneStepDataset, RolloutDataset, one_step_collate
from rollout import rollout
from utils_eval import (
    compute_one_step_metrics,
    compute_mse_n,
    compute_mse_full,
    compute_emd
)
from utils import (
    _combine_std,
    _read_metadata,
    compute_multi_step_loss,
    device,
    fix_seed,
    get_elapsed_time,
    get_kinematic_mask,
    print_args,
    Stats,
    INPUT_SEQUENCE_LENGTH,
    NUM_PARTICLE_TYPES,
    NUM_WORKERS
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


def eval(args: argparse.Namespace) -> None:
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
    # model init
    metadata = _read_metadata(data_path=f"datasets/{args.dataset}")
    model_kwargs = dict(
        latent_size=128,
        mlp_hidden_size=128,
        mlp_num_hidden_layers=2,
        num_message_passing_steps=args.message_passing_steps,
    )
    
    # init simulator
    simulator = _get_simulator(
        model_kwargs=model_kwargs,
        metadata=metadata,
        vel_noise_std=args.noise_std,
        acc_noise_std=args.noise_std,
        args=args
    )
    
    # load model checkpoint
    checkpoint_path = f'{args.model_path}/{args.dataset}/{args.gnn_type}/{args.loss_type}'
    checkpoint_file = None
    for file in os.listdir(checkpoint_path):
        if file.startswith(f'best_val_mse_pos{args.checkpoint}'):
            checkpoint_file = os.path.join(checkpoint_path, file)
            break
    
    if not checkpoint_file:
        raise ValueError("No checkpoint exists!")
    print(f"\nLoad checkpoint from: {checkpoint_file}")
    
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    simulator.load_state_dict(checkpoint['model_state_dict'])

    metrics = {
        'mse_acc_1': [],
        'mse_pos_1': [],
    }
    time_step = 0
    simulator.eval()

    # compute one-step metrics using OneStepDataset
    print("\nComputing one-step metrics...")
    one_step_dataset = OneStepDataset(args.dataset, args.eval_split)
    one_step_dataloader = DataLoader(
        one_step_dataset,
        collate_fn=one_step_collate,
        batch_size=args.batch_size,
        shuffle=False
    )

    with torch.no_grad():
        for features, labels in one_step_dataloader:
            labels = labels.to(device)
            for key in ['positions', 'particle_types', 'n_particles_per_example']:
                features[key] = features[key].to(device)
            if 'step_context' in features:
                features['step_context'] = features['step_context'].to(device)
            
            # compute MSE1 metrics -- no noise added
            mse_pos_1, mse_acc_1 = compute_one_step_metrics(
                simulator, features, labels
            )
            
            metrics['mse_acc_1'].append(mse_acc_1)
            metrics['mse_pos_1'].append(mse_pos_1)

            if time_step % 200 == 0:
                print(f"Step: {time_step} | "
                      f"MSE-acc 1: {mse_acc_1:.2e} | "
                      f"MSE-pos 1: {mse_pos_1:.2e}")
            time_step += 1

    # compute trajectory metrics using RolloutDataset
    if args.compute_all_metrics:
        print("\nComputing trajectory metrics...")
        rollout_dataset = RolloutDataset(args.dataset, args.eval_split)
        rollout_dataloader = DataLoader(
            rollout_dataset,
            collate_fn=one_step_collate,
            batch_size=1,
            shuffle=False
        )

        metrics.update({
            'mse_pos_20': [],
            'mse_acc_20': [],
            'mse_pos_full': [],
            'mse_acc_full': [],
            'emd': []
        })
        time_step = 0

        with torch.no_grad():
            for features, labels in rollout_dataloader:
                labels = labels.to(device)
                for key in ['positions', 'particle_types', 'n_particles_per_example']:
                    features[key] = features[key].to(device)
                if 'step_context' in features:
                    features['step_context'] = features['step_context'].to(device)

                # run rollout
                num_steps = metadata['sequence_length'] - INPUT_SEQUENCE_LENGTH
                rollout_op = rollout(simulator, features, num_steps)
                rollout_op['metadata'] = metadata
                
                # compute trajectory metrics
                mse_pos_20, mse_acc_20 = compute_mse_n(
                    simulator, features, rollout_op, n_frames=20
                )
                mse_pos_full, mse_acc_full = compute_mse_full(
                    simulator, features, rollout_op
                )
                emd = compute_emd(simulator, features, rollout_op)
                
                metrics['mse_pos_20'].append(mse_pos_20)
                metrics['mse_acc_20'].append(mse_acc_20)
                metrics['mse_pos_full'].append(mse_pos_full)
                metrics['mse_acc_full'].append(mse_acc_full)
                metrics['emd'].append(emd)

                print(f"Trajectory: {time_step}\n"
                        f"\tMSE-pos 20:   {mse_pos_20:.3e} | MSE-acc 20:   {mse_acc_20:.3e}\n"
                        f"\tMSE-pos full: {mse_pos_full:.3e} | MSE-acc full: {mse_acc_full:.3e}\n"
                        f"\tEMD: {emd:.3e}")
                time_step += 1

    # average metrics
    print("\nAverage metrics across all evaluations:")
    for key in metrics:
        avg_value = np.mean(metrics[key])
        print(f"{key}: {avg_value:.3e}")


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
    model_path = f'{args.model_path}/{args.dataset}/{args.gnn_type}/{args.loss_type}'
    output_path = f'{args.output_path}/{args.dataset}/{args.gnn_type}/{args.loss_type}/{args.eval_split}'
    os.makedirs(output_path, exist_ok=True)
    files = os.listdir(model_path)
    file_name = None

    for file in files:
        if file.startswith(f'best_val_mse_pos{args.checkpoint}'):
            file_name = os.path.join(model_path, file)
            break

    if not file_name:
        raise ValueError("No checkpoint exists!")
    else:
        print(f"\nLoad checkpoint from: {file_name}")

    checkpoint = torch.load(file_name, map_location=device, weights_only=False)
    simulator.load_state_dict(checkpoint['model_state_dict'])

    # evaluation loop
    mse_loss = F.mse_loss
    total_loss = []
    time_step = 0

    print("\nStarting rollout evaluation...")
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
            
            # compute MSE-pos losses
            loss_mse = mse_loss(
                rollout_op['predicted_rollout'], 
                rollout_op['ground_truth_rollout']
            )
            total_loss.append(loss_mse)
            print(f"Trajectory Rollout #{time_step} | MSE-pos: {loss_mse:.3e}")

            # save rollout results
            file_name = f'{time_step}.pkl'
            file_name = os.path.join(output_path, file_name)
            print(f"Saving rollout file {time_step}.\n")
            with open(file_name, 'wb') as file:
                pickle.dump(rollout_op, file)
            time_step += 1

        average_loss = torch.tensor(total_loss).mean().item()
        print(f"Average rollout loss is: {average_loss:.3e}")


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

    # data setup
    if args.loss_type == 'one_step':
        train_dataset = OneStepDataset(args.dataset, 'train')
        valid_dataset = OneStepDataset(args.dataset, 'valid')
    elif args.loss_type == 'multi_step':
        train_dataset = RolloutDataset(args.dataset, 'train')
        valid_dataset = RolloutDataset(args.dataset, 'valid')

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    # add val split to eval "one-step" MSE periodically
    num_valid_samples = len(valid_dataset)
    eval_subset_size = min(2000, num_valid_samples) # eval on ~2k samples
    validation_frequency = 3000

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

    # optimization setup
    optimizer = torch.optim.Adam(
        simulator.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # LR schedule setup (exp decay)
    decay_steps = int(2e5)
    min_lr = 1e-6
    warmup = int(1e4)

    # training state tracking
    best_val_pos_loss = float("inf")
    val_mse_pos_history = deque()
    val_mse_acc_history = deque()
    val_steps_history = deque()
    global_step = 0
    patience = 30
    steps_without_improvement = 0

    # checkpoint set up
    checkpoint_dir = f'{args.model_path}/{args.dataset}/{args.gnn_type}/{args.loss_type}'
    latest_checkpoint_path = os.path.join(checkpoint_dir, f'latest_checkpoint{args.checkpoint}.pt')
    best_model_path = os.path.join(checkpoint_dir, f'best_val_mse_pos{args.checkpoint}.pt')

    # helper function to save checkpoint
    def save_checkpoint(path):
        checkpoint = {
            'model_state_dict': simulator.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
            'val_mse_pos_history': list(val_mse_pos_history),
            'val_mse_acc_history': list(val_mse_acc_history),
            'val_steps_history': list(val_steps_history),
            'best_val_pos_loss': best_val_pos_loss,
            'steps_without_improvement': steps_without_improvement,
            'generator_state': generator.get_state(),
        }
        torch.save(checkpoint, path)
    
    # save if sbatch ends
    def save_checkpoint_on_signal(signum, frame):
        print(f"\nReceived signal {signum}. Saving checkpoint...")
        save_checkpoint(latest_checkpoint_path)
        print(f"Checkpoint saved at step {global_step}. Exiting.")
        exit(0)
    
    # signal handler for SIGTERM
    signal.signal(signal.SIGTERM, save_checkpoint_on_signal)

    # load from existing checkpoint, if available
    if os.path.exists(latest_checkpoint_path):
        print(f"\nLoading checkpoint from {latest_checkpoint_path}...")
        checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
        simulator.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        global_step = checkpoint['global_step']
        val_mse_pos_history = deque(checkpoint.get('val_mse_pos_history', []))
        val_mse_acc_history = deque(checkpoint.get('val_mse_acc_history', []))
        val_steps_history = deque(checkpoint.get('val_steps_history', []))
        best_val_pos_loss = checkpoint.get('best_val_pos_loss', best_val_pos_loss)
        steps_without_improvement = checkpoint.get('steps_without_improvement', 0)
        generator.set_state(checkpoint['generator_state'].cpu().type(torch.ByteTensor))
        print(f"Resumed training from step {global_step}.")

    # create DataLoader AFTER setting the generator's state
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=one_step_collate,
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        shuffle=True,
        generator=generator  # for deterministic shuffling
    )

    # helper function to compute loss for both training and val
    def compute_loss(features, target_next_position, mode: str = 'train'):
        """
        Compute loss for training or validation.

        Args:
            features: Batch features.
            target_next_position: Target next positions.
            mode: 'train' or 'val' to specify the mode.

        Returns:
            Tuple containing:
                - loss: The computed loss tensor (for training).
                - mse_acc: Mean squared error for acceleration.
                - mse_pos: Mean squared error for position.
        """
        non_kinematic_mask = torch.logical_not(get_kinematic_mask(features['particle_types']))
        if args.loss_type == 'one_step':
            if mode == 'train':
                # add noise with masking
                sampled_noise = get_random_walk_noise_for_position_sequence(
                    features['positions'],
                    noise_std_last_step=args.noise_std
                ).to(device)
                
                noise_mask = non_kinematic_mask.unsqueeze(1).unsqueeze(2)
                sampled_noise *= noise_mask
            else:
                # no noise during validation
                sampled_noise = torch.zeros_like(features['positions']).to(device)
                
            # forward pass and loss calculation
            pred_acceleration, target_acceleration = simulator.get_predicted_and_target_normalized_accelerations(
                next_position=target_next_position,
                position_sequence=features['positions'],
                position_sequence_noise=sampled_noise,
                n_particles_per_example=features['n_particles_per_example'],
                particle_types=features['particle_types'],
                global_context=features.get('step_context')
            )

            # compute MSE-acc on non-kinematic particles
            loss = (pred_acceleration[non_kinematic_mask] - target_acceleration[non_kinematic_mask]) ** 2
            loss = torch.mean(loss)
            mse_acc = loss.item()

            # compute MSE-pos
            simulator.eval()
            with torch.no_grad():
                predicted_next_position = simulator(
                    position_sequence=features['positions'],
                    n_particles_per_example=features['n_particles_per_example'],
                    particle_types=features['particle_types'],
                    global_context=features.get('step_context')
                )
                mse_pos = F.mse_loss(predicted_next_position, target_next_position).item()
            if mode == 'train':
                simulator.train()

            return loss, mse_acc, mse_pos

        elif args.loss_type == 'multi_step':
            # ground truth positions
            current_positions = features['positions'][:, :INPUT_SEQUENCE_LENGTH]
            pred_accelerations = []
            target_accelerations = []
            
            # use model's OWN preds
            for step in range(args.n_rollout_steps + 1):
                # ground-truth next position at this step
                target_position = features['positions'][:, INPUT_SEQUENCE_LENGTH + step]
                position_noise = torch.zeros_like(current_positions)
                
                # pred & tgt acc
                pred_accel, tgt_accel = simulator.get_predicted_and_target_normalized_accelerations(
                    next_position=target_position,
                    position_sequence=current_positions,
                    position_sequence_noise=position_noise,  # no added noise
                    n_particles_per_example=features['n_particles_per_example'],
                    particle_types=features['particle_types'],
                    global_context=features.get('step_context')
                )
                pred_accelerations.append(pred_accel)
                target_accelerations.append(tgt_accel)

                # pred next pos
                predicted_next_position = simulator(
                    position_sequence=current_positions,
                    n_particles_per_example=features['n_particles_per_example'],
                    particle_types=features['particle_types'],
                    global_context=features.get('step_context')
                )

                # update pos sequence with predicted pos
                current_positions = torch.cat([
                    current_positions[:, 1:],
                    predicted_next_position.unsqueeze(1)
                ], dim=1)
            
            # avg MSE-acc across steps
            loss = compute_multi_step_loss(
                pred_accelerations,
                target_accelerations,
                non_kinematic_mask
            )
            mse_acc = loss.item()

            # compute MSE-pos (FINAL predicted_next_position vs ground-truth)
            final_ground_truth_pos = features['positions'][:, INPUT_SEQUENCE_LENGTH + args.n_rollout_steps]
            mse_pos = F.mse_loss(predicted_next_position, final_ground_truth_pos).item()

            return loss, mse_acc, mse_pos

    start_time = time.time()
    print(f"\nTraining for {args.num_steps} steps")
    print(f"{len(train_dataloader)} steps per epoch")
    print(f"Validating every {validation_frequency} steps")
    print(f"Number of Workers: {NUM_WORKERS}")
    print(f"Device: {device}")
    print("\nStarting training...")
    try:
        while global_step < args.num_steps:
            for features, target_next_position in train_dataloader:
                if global_step >= args.num_steps:
                    break
                
                target_next_position = target_next_position.to(device)
                for key in ['positions', 'particle_types', 'n_particles_per_example']:
                    features[key] = features[key].to(device)
                if 'step_context' in features:
                    features['step_context'] = features['step_context'].to(device)

                simulator.train()
                optimizer.zero_grad()

                loss, train_mse_acc, train_mse_pos = compute_loss(features, target_next_position, mode='train')

                # optimization step
                loss.backward()
                clip_grad_norm_(simulator.parameters(), 1.0)
                optimizer.step()

                # handle LR over time
                if global_step < warmup:
                    # linear warmup
                    current_lr = args.lr * (global_step / warmup)
                else:
                    # exponential decay after warmup
                    decay_progress = (global_step - warmup) / (decay_steps - warmup)
                    current_lr = (args.lr - min_lr) * (0.1 ** decay_progress) + min_lr

                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                
                # logging
                if global_step % 250 == 0:
                    print(f"{get_elapsed_time(start_time)} "
                          f"Step: {global_step} | "
                          f"MSE-acc: {train_mse_acc:.2e} | "
                          f"MSE-pos: {train_mse_pos:.2e} | "
                          f"LR: {current_lr:.3e}")

                # validation loop
                if global_step % validation_frequency == 0 and global_step > 0:
                    simulator.eval()
                    valid_indices = np.random.choice(num_valid_samples, eval_subset_size, replace=False)
                    valid_subset = Subset(valid_dataset, valid_indices)
                    val_dataloader = DataLoader(
                        valid_subset, 
                        collate_fn=one_step_collate,
                        batch_size=args.batch_size,
                        num_workers=NUM_WORKERS,
                        pin_memory=True,
                        shuffle=True
                    )

                    val_losses = []
                    with torch.no_grad():
                        for vfeat, vtarget in val_dataloader:
                            vtarget = vtarget.to(device)
                            for key in ['positions', 'particle_types', 'n_particles_per_example']:
                                vfeat[key] = vfeat[key].to(device)
                            if 'step_context' in vfeat:
                                vfeat['step_context'] = vfeat['step_context'].to(device)

                            # Compute validation loss
                            val_loss, val_mse_acc, val_mse_pos = compute_loss(vfeat, vtarget, mode='val')
                            val_losses.append({
                                'accel': val_mse_acc,
                                'pos': val_mse_pos
                            })

                    mean_val_acc_loss = float(np.mean([x['accel'] for x in val_losses]))
                    mean_val_pos_loss = float(np.mean([x['pos'] for x in val_losses]))
                    val_mse_pos_history.append(mean_val_pos_loss) # store val MSE-pos history
                    val_mse_acc_history.append(mean_val_acc_loss) # store val MSE-acc history
                    val_steps_history.append(global_step)
                    print(f"{get_elapsed_time(start_time)} Validation MSE-acc: {mean_val_acc_loss:.2e} | "
                          f"MSE-pos: {mean_val_pos_loss:.2e}")

                    # check if new best MSE-pos
                    if mean_val_pos_loss < best_val_pos_loss:
                        best_val_pos_loss = mean_val_pos_loss
                        steps_without_improvement = 0
                        print(f"\t\tNew best validation MSE-pos -> Saving model...")
                        save_checkpoint(best_model_path)
                    else:
                        steps_without_improvement += 1
                        print(f"\t\tNo improvement. Steps without improvement: {steps_without_improvement}/{patience}")

                    save_checkpoint(latest_checkpoint_path)
                    print(f"{get_elapsed_time(start_time)} Checkpoint saved at step {global_step}.")

                    # early stopping
                    if steps_without_improvement >= patience:
                        print("\nEarly stopping triggered - No improvement on validation set.")
                        raise KeyboardInterrupt
                
                global_step += 1
    
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        print(f"Saving current checkpoint - Step: {global_step}")
    
    print(f"\n{get_elapsed_time(start_time)} Training completed!")
    print(f"Best validation MSE-pos achieved: {best_val_pos_loss:.2e}")
    print(f"Best model saved at: {best_model_path}")
    save_checkpoint(latest_checkpoint_path)


def parse_arguments():
    """Parse command line arguments organized by usage mode."""
    parser = argparse.ArgumentParser(description="Learning to Simulate.")
    
    # global args (used by all modes)
    global_group = parser.add_argument_group('Global Arguments')
    global_group.add_argument('--mode',
                              default='train',
                              choices=['train', 'eval', 'eval_rollout'],
                              help='Operation mode: train model, evaluate, or rollout evaluation.')
    global_group.add_argument('--dataset',
                              default="Water",
                              type=str,
                              help='Dataset directory.')
    global_group.add_argument('--model_path',
                              default="models",
                              type=str,
                              help='Top-level directory for saving models.')
    global_group.add_argument('--checkpoint',
                              default=2,
                              type=int,
                              help='Checkpoint ID to save/load model training.')
    global_group.add_argument('--gnn_type',
                              default='interaction_net',
                              choices=['interaction_net', 'gat', 'gcn'],
                              help='Type of GNN to use as processor.')
    global_group.add_argument('--batch_size',
                              default=2,
                              type=int,
                              help='Batch size for training and evaluation.')
    global_group.add_argument('--message_passing_steps',
                              default=10,
                              type=int,
                              help='Number of GNN message passing steps.')
    global_group.add_argument('--noise_std',
                              default=0.0003,
                              type=float,
                              help='Standard deviation of noise for training and evaluation.')
    
    # training-specific args
    train_group = parser.add_argument_group('Training Arguments (only used when mode=train)')
    train_group.add_argument('--num_steps',
                             default=2e7,
                             type=int,
                             help='Maximum number of training steps.')
    train_group.add_argument('--seed',
                             type=int,
                             default=483,
                             help='Random seed for reproducibility.')
    train_group.add_argument('--lr',
                             type=float,
                             default=1e-4,
                             help='Initial learning rate.')
    train_group.add_argument('--weight_decay',
                             type=float,
                             default=0,
                             help='Weight decay for optimizer.')
    train_group.add_argument('--loss_type',
                             default='one_step',
                             choices=['one_step', 'multi_step'],
                             help='Type of loss function to use during training')
    train_group.add_argument('--n_rollout_steps',
                             type=int,
                             default=1,
                             help='Number of rollout steps for multi-step loss')
    
    # eval-specific args
    eval_group = parser.add_argument_group('Evaluation Arguments (only used when mode=eval or eval_rollout)')
    eval_group.add_argument('--eval_split',
                            default='test',
                            choices=['train', 'valid', 'test'],
                            help='Dataset split to use for evaluation.')
    eval_group.add_argument('--output_path',
                            default="rollouts",
                            type=str,
                            help='Path for saving rollout results (only used in eval_rollout mode).')
    eval_group.add_argument('--compute_all_metrics',
                            action='store_true',
                            help='Computes all metrics (MSE1, MSE20, MSE400, EMD) instead of just MSE1.')
    
    # GAT-specific arguments
    gat_group = parser.add_argument_group('GAT-specific Arguments (only used when gnn_type=gat)')
    gat_group.add_argument('--gat_heads',
                           type=int,
                           default=8,
                           help='Number of attention heads.')
    gat_group.add_argument('--out_heads',
                           type=int,
                           default=1,
                           help='Number of output heads.')
    
    # GNN architecture args
    gnn_group = parser.add_argument_group('GNN Architecture Arguments')
    gnn_group.add_argument('--hidden_channels',
                           type=int,
                           default=32,
                           help='Number of hidden channels in GNN layers.')
    gnn_group.add_argument('--dropout',
                           type=float,
                           default=0.0,
                           help='Dropout rate.')
    gnn_group.add_argument('--num_gnn_layers',
                           type=int,
                           default=2,
                           help='Number of GNN layers.')
    gnn_group.add_argument('--use_bn',
                           action='store_true',
                           help='Use Batch Normalization in GNN layers.')
    gnn_group.add_argument('--use_residual',
                           action='store_true',
                           help='Use residual connections in GNN layers (only applicable for GCN).')
    
    args = parser.parse_args()
    
    # create output dirs if don't exist
    os.makedirs(f'{args.model_path}/{args.dataset}/{args.gnn_type}/{args.loss_type}',
               exist_ok=True)
    if args.mode == 'eval_rollout':
        os.makedirs(f'{args.output_path}/{args.dataset}/{args.gnn_type}/{args.loss_type}/{args.eval_split}',
                   exist_ok=True)
    
    print_args(args)
    return args


if __name__ == '__main__':
    args = parse_arguments()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        eval(args)
    elif args.mode == 'eval_rollout':
        eval_rollout(args)
    else:
        raise ValueError("Unrecognized mode!")
