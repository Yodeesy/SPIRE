# args.py
import argparse
import os

parser = argparse.ArgumentParser(description="SPIRE: One-Shot Federated Graph Learning")

# ==============================================================================
# 1. Environment & Paths
# ==============================================================================
current_path = os.path.abspath(__file__)
dataset_path = os.path.join(os.path.dirname(current_path), 'datasets')
root_dir = os.path.join(dataset_path, 'raw_data')

if not os.path.exists(root_dir):
    os.makedirs(root_dir, exist_ok=True)

log_path = os.path.join(os.path.dirname(current_path), 'logs')
if not os.path.exists(log_path):
    os.makedirs(log_path, exist_ok=True)

env_group = parser.add_argument_group('Environment')
env_group.add_argument("--dataset", type=str, default="cora", help="Dataset name: cora, ogbn-arxiv, etc.")
env_group.add_argument("--dataset_dir", type=str, default=root_dir, help="Root path for datasets")
env_group.add_argument("--logs_dir", type=str, default=log_path, help="Path to save logs")
env_group.add_argument("--device_id", type=int, default=0, help="GPU device ID")
env_group.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

# ==============================================================================
# 2. Data Partitioning (Non-IID Settings)
# ==============================================================================
data_group = parser.add_argument_group('Data Partitioning')
data_group.add_argument("--task", type=str, default="node_classification")
data_group.add_argument("--num_clients", type=int, default=10, help="Number of clients")
data_group.add_argument("--dirichlet_beta", type=float, default=0.05, help="Dirichlet Beta (smaller = more Non-IID)")
data_group.add_argument("--least_samples", type=int, default=10, help="Min samples per client")
data_group.add_argument("--dataset_split_metric", type=str, default="transductive", choices=["transductive"])
data_group.add_argument("--train_val_test_split", type=float, nargs='+', default=[0.2, 0.4, 0.4])
data_group.add_argument("--dirichlet_try_cnt", type=int, default=10000)

# ==============================================================================
# 3. Federated Learning Settings
# ==============================================================================
fl_group = parser.add_argument_group('Federated Learning')
fl_group.add_argument("--fed_algorithm", type=str, default="SPIRE")
fl_group.add_argument("--num_rounds", type=int, default=1, help="Total communication rounds (Set to 1 for One-Shot)")
fl_group.add_argument("--cl_sample_rate", type=float, default=1.0, help="Client sampling rate")
fl_group.add_argument("--T_L", type=int, default=1, help="Local epochs (Skipped in One-Shot SPIRE)")

# ==============================================================================
# 4. Global Model (Backbone GNN)
# ==============================================================================
model_group = parser.add_argument_group('Backbone Model')
model_group.add_argument("--model", type=str, default="GCN", help="GNN Backbone: GCN, GAT, SAGE")
model_group.add_argument("--hidden_dim", type=int, default=256)
model_group.add_argument("--num_layers", type=int, default=2)
model_group.add_argument("--dropout", type=float, default=0.3)
model_group.add_argument("--learning_rate", type=float, default=0.01, help="Local learning rate for client training (FedAvg/FedProx)")
model_group.add_argument("--server_lr", type=float, default=0.005, help="Learning rate for Global GNN")
model_group.add_argument("--weight_decay", type=float, default=0.0005)
model_group.add_argument('--patience', type=int, default=100, help='Patience rounds for early stopping based on Val Acc')
model_group.add_argument("--global_batch_size", type=int, default=4096, help="Batch size for global model training via NeighborLoader")

# ==============================================================================
# 5. SPIRE Generator (Diffusion Model)
# ==============================================================================
sea_group = parser.add_argument_group('SPIRE Generator')
sea_group.add_argument("--gen_mode", type=str, default="server", help="Generator location: server or federated")
sea_group.add_argument("--gen_train_steps", type=int, default=30, help="Generator training steps per round")
sea_group.add_argument("--gen_lr", type=float, default=0.0001, help="Learning rate for Generator")
sea_group.add_argument("--gen_hidden", type=int, default=256, help="Hidden dim for diffusion network")
sea_group.add_argument("--noise_dim", type=int, default=64, help="Dimension of input noise z")
sea_group.add_argument("--diff_steps", type=int, default=8, help="Number of denoising steps (T)")
sea_group.add_argument("--gen_knn", type=int, default=6, help="k for KNN structure generation")
sea_group.add_argument("--max_jitter", type=float, default=0.03, help="Feature-level gaussian noise std (Output Jitter).")
sea_group.add_argument("--gen_batch_size", type=int, default=4096, help="Batch size for generator training")

# ==============================================================================
# 6. SPIRE Loss Weights & Hyperparams
# ==============================================================================
loss_group = parser.add_argument_group('SPIRE Losses')
loss_group.add_argument("--server_epochs", type=int, default=300, help="Server-side generation & distillation epochs (T_S)")
loss_group.add_argument("--T_G", type=int, default=30, help="Global GNN training steps per round")
loss_group.add_argument("--tau", type=float, default=1.0, help="Temperature for SWA weights (Entropy-based)")

# --- Generator Optimization (G) ---
# 1. Barycenter Matching Loss
loss_group.add_argument("--lambda_align", type=float, default=0.1, help="Weight for Barycenter Matching Loss (MSE vs Dynamic b)")
# 2. Structural Entropy Loss
loss_group.add_argument("--lambda_se", type=float, default=1e-5, help="Weight for Structural Entropy Loss (with Warm-up)")

# Parse
args = parser.parse_args()