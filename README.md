# Learning to Simulate Physics

This project provides an implementation for learning to simulate particle-based physics, adapted from [Learn-to-Simulate](https://github.com/Emiyalzn/Learn-to-Simulate/tree/main).

## Overview

The system learns to simulate particle-based physics by using Graph Neural Networks (GNNs) to model particle interactions. It supports different types of simulations including water drops, sand dynamics, and goop behavior.

## Features

- Multiple GNN architectures (e.g. interaction networks and GAT)
- Single-step and rollout predictions
- Support for different particle types
- Noise injection for robustness
- Checkpoint saving and loading
- Validation during training

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Elder453/CPSC483-final.git
cd CPSC483-final
```

2. Install dependencies:
```bash
conda env create -f environment.yml
```
  **Note**: CUDA-specific versions of PyTorch and TensorFlow may need to be installed manually depending on your system’s GPU and CUDA version. Refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/) and [TensorFlow](https://www.tensorflow.org/install) installation guide for details.

3. Create necessary directories:
```bash
mkdir -p ./datasets
mkdir -p ./models
mkdir -p ./rollouts
```

## Dataset

1. Download a dataset (e.g., Water or WaterDropSample):
```bash
bash ./download_dataset.sh WaterDropSample ./datasets
```

## Usage

### Training

Train a model with default parameters:
```bash
bash ./train.sh
```

### Evaluation

Evaluate model predictions:
```bash
bash ./eval.sh
```

Generate and animate trajectory rollouts:
```bash
bash ./rollout.sh
```

## Key Parameters

Within each `bash` script, there are various parameters that may be tuned or customized for usage. For convenience, we have provided a bash script for each of the core functionalities with default behavior; please modify those scripts directly or use the command line for customization deviating from the default behavior (`train.sh`, `eval.sh`, `rollout.sh`).

- `--mode`: Training or evaluation mode (`train`, `eval`, `eval_rollout`)
- `--dataset`: Name of dataset to use
- `--batch_size`: Number of samples per batch
- `--num_steps`: Number of training steps
- `--message_passing_steps`: Number of GNN message passing steps
- `--gnn_type`: Type of GNN to use (`interaction_net`, `gcn`, `gat`)
- `--noise_std`: Standard deviation of noise injection
- `--lr`: Learning rate

## Directory Structure

```
/CPSC483-final/
├── datasets/           # Dataset storage
│   └── WaterDropSample/
│       ├── train/
│       ├── valid/
│       └── test/
├── models/             # Saved model checkpoints
│   └── WaterDropSample/
├── rollouts/           # Generated rollouts
│   └── WaterDropSample/
├── src/
│   ├── main.py
│   ├── dataloader.py
│   ├── graph_network.py
│   ├── learned_simulator.py
│   ├── message_passing.py
│   ├── models.py
│   ├── render_rollout.py
│   ├── rollout.py
│   ├── utils_connectivity.py
│   ├── utils_eval.py
│   ├── utils_noise.py
│   └── utils.py
├── environment.yml
├── transform_to_pkl.py
├── download_dataset.sh
├── train.sh
├── eval.sh
├── rollout.sh
└── README.md
```

## Notes

- The code assumes CUDA availability. For CPU-only usage, modify device settings and `environment.yml` accordingly.
- Training time varies based on dataset size and computational resources.
- Best validation checkpoints are automatically saved during training.
- Parameters may need adjustment based on your specific use case.

## Acknowledgments

The original TensorFlow implementation was developed by [DeepMind](https://github.com/deepmind/deepmind-research) and published at ICML 2020. The research paper is titled “Learning to Simulate Complex Physics with Graph Networks” and can be accessed on [arXiv](https://arxiv.org/abs/2002.09405).

This PyTorch implementation builds upon [Learn-to-Simulate](https://github.com/Emiyalzn/Learn-to-Simulate/tree/main), which translated the original TensorFlow codebase into PyTorch. We extend our gratitude to the authors for their foundational work.

```shell
@article
{DBLP:journals/corr/abs-2002-09405,
  author    = {Alvaro Sanchez{-}Gonzalez and
               Jonathan Godwin and
               Tobias Pfaff and
               Rex Ying and
               Jure Leskovec and
               Peter W. Battaglia},
  title     = {Learning to Simulate Complex Physics with Graph Networks},
  journal   = {CoRR},
  volume    = {abs/2002.09405},
  year      = {2020},
  url       = {https://arxiv.org/abs/2002.09405},
  eprinttype = {arXiv},
  eprint    = {2002.09405},
  timestamp = {Mon, 02 Mar 2020 16:46:06 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2002-09405.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
