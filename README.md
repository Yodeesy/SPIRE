# Structural Entropy-Driven Graph Diffusion Generation for One-Shot Federated Graph Learning

[![arXiv](https://img.shields.io/badge/arXiv-2609.06499-b31b1b.svg)](https://arxiv.org/abs/2609.06499)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Abstract

One-shot federated graph learning (FGL) requires the server to estimate client contributions from highly compressed information, yet conventional volume-based weighting captures the amount of client data while overlooking how its connectivity is organized. In this paper, we propose SPIRE, a Structural Entropy-Driven Graph Diffusion Generation method that introduces topology-aware client differentiation into one-shot FGL. Specifically, we employ first-order degree-distribution structural entropy as a compact descriptor of degree-mass dispersion and use it to derive structural client weights, providing an inductive bias that accounts for differences in graph topology beyond data volume. On the generation side, a graph diffusion model on the server synthesizes pseudographs conditioned on the weighted client prototypes, capturing both semantic and structural information without requiring additional client-side training. The generated pseudographs are then assembled via disjoint union fusion to train a global graph neural network. Extensive experiments on seven real-world graph datasets demonstrate that SPIRE consistently outperforms conventional and one-shot FGL methods, with particularly strong gains under highly heterogeneous (non-IID) and graph-perturbed settings.

----

## Highlights

* **True One-Shot Communication**: Accomplishes federated training in a single communication round. Clients upload only class prototypes and an $\mathcal{O}(1)$ scalar entropy score ($\sim$0.38 MB total), completely bypassing multi-round bandwidth bottlenecks.
* **Topology-Aware Structural Prior**: Employs first-order structural entropy to dynamically balance client contributions under extreme non-IID skew without inspecting private local edge topologies.
* **Data-Free Graph Diffusion**: Reconstructs global graph semantics via trajectory-guided conditional diffusion and Wasserstein barycenters, physically insulating the server from local structural corruption.

---

## 1. Requirements

The experiments were conducted using:
- **Python** >= 3.9
- **PyTorch** == 2.4.1
- **PyTorch Geometric (PyG)** == 2.6.1
- A CUDA-enabled NVIDIA GPU

### Environment Setup

1. Install **PyTorch** according to your local CUDA version (refer to the [PyTorch official installation guide](https://pytorch.org/get-started/locally/)):

      For example, for CUDA 12.1:

   ```bash
   pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

2. Install the remaining dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   

## 2. Quick Start

All benchmark datasets (e.g., `Cora`, `CiteSeer`, `PubMed`, `ogbn-arxiv`) will be automatically downloaded and processed upon the first run (stored under `./datasets/raw_data`).

### Running Experiments

- **Run on Cora (Example):**

  ```Bash
  python main.py --fed_algorithm SPIRE --dataset cora --num_clients 10 --dirichlet_beta 0.05
  ```
  
- **Run via Script:**

  ```Bash
  chmod +x run.sh
  ./run.sh
  ```

> **Note:** The final global test accuracy and logs will be printed in the console and automatically saved to `./logs/`.

### Key Arguments

- `--dataset`: Dataset name (`cora`, `citeseer`, `pubmed`, `ogbn-arxiv`, etc.).
- `--num_clients`: Number of federated clients.
- `--dirichlet_beta`: Concentration parameter for Dirichlet Non-IID distribution (smaller values indicate higher data heterogeneity).

## 3. Project Structure

Plaintext

```txt
.
├── args.py                  # Command-line arguments and configuration
├── main.py                  # Main execution entry
├── run.sh                   # Shell script for batch experiments
├── requirements.txt         # Environment dependencies
├── README.md
├── algorithm/
│   ├── Base.py              # Base class for federated algorithms
│   └── SPIRE.py             # Implementation of the proposed method
├── backbone/
│   ├── GCN.py               # GNN backbone architectures
│   └── GraphDiffusionGenerator.py  # Graph diffusion generator
├── datasets/
│   ├── dataset_loader.py    # Standard dataset loaders
│   ├── dataset_ds_loader.py # Downstream dataset loaders
│   └── partition.py         # Non-IID graph partitioning strategy
└── utils/
    ├── GW_utils.py          # Gromov-Wasserstein utilities
    ├── ot_align.py          # Optimal transport alignment
    ├── ot_utils.py          # OT helper functions
    ├── graph_ops.py         # Graph operation utilities
    ├── stats.py             # Metrics and logging statistics
    ├── taskflow.py          # Pipeline flow management
    ├── training_utils.py    # Training helper functions
    ├── set_seed.py          # Reproducibility seed setting
    └── logger.py            # Logger utilities
```

## Citation
If you find this work or code useful in your research, please consider citing:
```text
@misc{zheng2026structuralentropydrivengraphdiffusion,
      title={Structural Entropy-Driven Graph Diffusion Generation for One-Shot Federated Graph Learning}, 
      author={Shutong Zheng and Lele Fu and Sheng Huang and Wei Yang Bryan Lim and Chuan Chen},
      year={2026},
      eprint={2609.06499},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2609.06499}, 
}
```
