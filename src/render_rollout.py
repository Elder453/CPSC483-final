import pickle
import argparse
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

TYPE_TO_COLOR = {
    3: "black",   # Boundary particles.
    0: "green",   # Rigid solids.
    7: "magenta", # Goop.
    6: "gold",    # Sand.
    5: "blue",    # Water.
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Render all rollouts in directory.")
    parser.add_argument(
        '--output_path',
        default="rollouts",
        type=str,
        help='Base output directory for rollouts.'
    )
    parser.add_argument(
        '--dataset',
        required=True,
        type=str,
        help='Name of the dataset used in training.'
    )
    parser.add_argument(
        '--gnn_type',
        required=True,
        type=str,
        choices=['gcn', 'gat', 'trans_gnn', 'interaction_net'],
        help='Type of GNN used.'
    )
    parser.add_argument(
        '--loss_type',
        required=True,
        type=str,
        choices=['one_step', 'multi_step'],
        help='Loss type used in training.'
    )
    parser.add_argument(
        '--eval_split',
        default='test',
        choices=['train', 'valid', 'test'],
        help='Dataset split to use for evaluation.'
    )
    parser.add_argument(
        '--step_stride',
        default=3,
        type=int,
        help='Stride of steps to skip.'
    )

    args = parser.parse_args()
    args.rollout_path = f"{args.output_path}/{args.dataset}/{args.gnn_type}/{args.loss_type}/{args.eval_split}"
    return args


def render_rollout(rollout_data, output_path, step_stride=3):
    """Render a single rollout pickle file to gif.
    
    Args:
        rollout_data: Dictionary containing rollout data
        output_path: Path where to save the gif
        step_stride: Number of steps to skip between frames
        fps: Frames per second in output gif
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    plot_info = []
    for ax_i, (label, rollout_field) in enumerate(
        [("Ground Truth", "ground_truth_rollout"),
         ("Prediction", "predicted_rollout")]):
        # append init positions to get full trajectory
        trajectory = np.concatenate([
            rollout_data["initial_positions"].cpu(),
            rollout_data[rollout_field].cpu()], axis=0) # [time_steps, num_points, num_dimensions]
        ax = axes[ax_i]
        ax.set_title(label)
        bounds = rollout_data["metadata"]["bounds"]
        ax.set_xlim(bounds[0][0], bounds[0][1])
        ax.set_ylim(bounds[1][0], bounds[1][1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1.)
        points = {
            particle_type: ax.plot([], [], "o", ms=2, color=color)[0]
            for particle_type, color in TYPE_TO_COLOR.items()}
        plot_info.append((ax, trajectory, points))

    num_steps = trajectory.shape[0]

    def update(step_i):
        outputs = []
        for _, trajectory, points in plot_info:
            for particle_type, line in points.items():
                mask = rollout_data["particle_types"].cpu() == particle_type
                line.set_data(trajectory[step_i, mask, 0],
                              trajectory[step_i, mask, 1])
                outputs.append(line)
        return outputs

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=np.arange(0, num_steps, step_stride),
        interval=10
    )
    anim.save(output_path, writer='imagemagick', fps=10)
    plt.close(fig)


def main(args):
    if not os.path.exists(args.rollout_path):
        raise ValueError(f"Rollout path does not exist: {args.rollout_path}")
        
    # get .pkl files in dir
    pkl_files = [f for f in os.listdir(args.rollout_path) if f.endswith('.pkl')]
    if not pkl_files:
        raise ValueError(f"No pickle files found in {args.rollout_path}")
        
    print(f"\nFound {len(pkl_files)} pickle files to process")
    
    # process each .pkl file
    for pkl_file in tqdm(pkl_files, desc="Converting rollouts to gifs"):
        pkl_path = os.path.join(args.rollout_path, pkl_file)
        gif_path = os.path.join(args.rollout_path, pkl_file.replace('.pkl', '.gif'))
        
        # skip if gif already exists
        if os.path.exists(gif_path):
            continue
            
        try:
            with open(pkl_path, 'rb') as f:
                rollout_data = pickle.load(f)
            render_rollout(
                rollout_data,
                gif_path,
                step_stride=args.step_stride,
            )
        except Exception as e:
            print(f"\nError processing {pkl_file}: {str(e)}")


if __name__ == '__main__':
    main(parse_arguments())
