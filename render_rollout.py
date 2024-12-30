import pickle
import argparse
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

TYPE_TO_COLOR = {
    3: "black",   # Boundary particles.
    0: "green",   # Rigid solids.
    7: "magenta", # Goop.
    6: "gold",    # Sand.
    5: "blue",    # Water.
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Render rollout.")
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
        '--time_step',
        default=0,
        type=int,
        help='ID for rollout pkl file.'
    )
    parser.add_argument(
        '--step_stride',
        default=3,
        type=int,
        help='Stride of steps to skip.'
    )
    parser.add_argument(
        '--block_on_show',
        default=True,
        type=bool,
        help='For test purposes.'
    )

    args = parser.parse_args()
    args.rollout_path = f"{args.output_path}/{args.dataset}/{args.gnn_type}/{args.loss_type}/{args.eval_split}"
    return args


def main(args):
    if not args.rollout_path:
        raise ValueError("A `rollout_path` must be passed.")
    with open(f"{args.rollout_path}/{args.time_step}.pkl", 'rb') as file:
        rollout_data = pickle.load(file)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    plot_info = []
    for ax_i, (label, rollout_field) in enumerate(
        [("Ground Truth", "ground_truth_rollout"),
         ("Prediction", "predicted_rollout")]):
        # Append the initial positions to get the full trajectory.
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
        frames=np.arange(0, num_steps, args.step_stride),
        interval=10
    )
    anim.save(f"{args.rollout_path}/{args.time_step}.gif", writer='imagemagick', fps=10)
    # plt.show(block=args.block_on_show)


if __name__ == '__main__':
    main(parse_arguments())
