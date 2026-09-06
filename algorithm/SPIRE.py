# algorithm/SPIRE.py
import numpy as np
import torch
import copy
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_scipy_sparse_matrix
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import knn_graph

from algorithm.Base import BaseServer, BaseClient
from utils.stats import graph_structural_entropy, compute_class_prototypes
from utils.graph_ops import fuse_graphs, softmax_weights
from backbone.GraphDiffusionGenerator import GraphDiffusionGenerator
from utils.training_utils import calculate_generator_loss
from utils.ot_utils import LearnableBarycenter, compute_sinkhorn_loss
from utils.utils import get_entropy_guided_samples

class SPIREServer(BaseServer):
    """
    SPIRE Server:
    1. Collects stats (Prototypes, Entropy) from clients.
    2. Trains a server-side Diffusion Generator.
    3. Generates synthetic subgraphs.
    4. Fuses them into a global mega-graph (Union).
    5. Trains the global GNN on this union graph.
    """

    def __init__(self, args, clients, model, data, logger):
        super(SPIREServer, self).__init__(args, clients, model, data, logger)
        self.args = args
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        self.feature_dim = self.data.x.shape[-1]
        self.num_classes = args.num_classes
        self.noise_dim = args.noise_dim

        # Get total server-side training epochs (T_S)
        self.server_epochs = getattr(self.args, "server_epochs", 200)

        # Optimizer for the global GNN
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=getattr(args, "server_lr", 0.01),
            weight_decay=getattr(args, "weight_decay", 5e-4)
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.server_epochs,
            eta_min=1e-5
        )

        # Hyperparameters
        self.tau = getattr(args, "tau", 1.0)  # Temp for entropy weights
        self.T_G = getattr(args, "T_G", 10)  # Global epochs per round

        # Initialize Server-side Generator
        # gen_mode can be 'server' (default) or 'federated'
        self.gen_mode = getattr(args, "gen_mode", "server")

        self.SPIRE_generator = GraphDiffusionGenerator(
            self.device,
            self.noise_dim,
            self.feature_dim,
            self.num_classes,
            num_timesteps=getattr(args, "diff_steps", 100),
            hidden_dim=getattr(args, "gen_hidden", 256),
            args=args
        ).to(self.device)

        self.ema_generator = copy.deepcopy(self.SPIRE_generator)

        for param in self.ema_generator.parameters():
            param.requires_grad = False

        self.gen_optimizer = torch.optim.Adam(
            self.SPIRE_generator.parameters(),
            lr=getattr(args, "gen_lr", 1e-3)
        )

        self.round_idx = 0
        self.global_state = None
        self.early_stop_patience = args.patience
        self.early_stop_counter = 0

        # Placeholder for dynamic barycenter optimizer
        self.b_learner = None
        self.b_optimizer = None

    def _one_shot_global_init(self):
        """
        Private method: Performs the heavy lifting of One-Shot aggregation.
        Calculates Entropy, Alphas, SWA Prototypes, and Optimal Transport.
        """
        print("  [SPIRE] One-Shot Communication: Initializing Global View...")
        sampled = list(self.sampled_clients)

        stats_list = []
        entropies = []
        client_protos = []
        client_num_samples = []

        # 1. Collect Raw Stats
        for cid in sampled:
            stats = self.clients[cid].get_stats()
            entropies.append(float(stats.get("S_k", 0.0)))

            n_k = float(stats.get("num_nodes", 1.0))
            client_num_samples.append(n_k)

            prot = stats.get("prototypes", None)
            if prot is not None and isinstance(prot, torch.Tensor):
                prot = prot.detach().cpu().numpy()
            client_protos.append(prot)
            stats_list.append(stats)

        if not stats_list: return None

        # 2. Entropy-based Weights
        alphas = softmax_weights(entropies, tau=self.tau)
        alphas = [float(a) for a in alphas]
        print(f"  [SPIRE] Entropies: {[f'{e:.2f}' for e in entropies]}")
        # 3. Adaptive Sampling Quantities
        adaptive_ns = get_entropy_guided_samples(stats_list, alphas, eta=1.0)

        # 4. SWA Prototypes
        global_protos_swa = np.zeros((self.num_classes, self.feature_dim), dtype=np.float32)
        total_w = 0.0
        for w, p in zip(alphas, client_protos):
            if p is not None:
                global_protos_swa += w * np.array(p)
                total_w += w
        if total_w > 0: global_protos_swa /= total_w

        # 5. OT Alignment
        ot_target = None
        try:
            from utils.ot_align import wasserstein_prototype_alignment
            ot_np = wasserstein_prototype_alignment(client_protos, alphas, self.num_classes, self.feature_dim)
            if ot_np is not None:
                ot_target = torch.tensor(ot_np, dtype=torch.float32, device=self.device)
                print("  [SPIRE] OT Alignment Calculated.")
        except:
            pass

        print("  [SPIRE] Initialization Complete. Stats Cached.")

        # Return a structured dictionary (The "Context")
        return {
            "stats_list": stats_list,
            "alphas": alphas,
            "adaptive_ns": adaptive_ns,
            "global_protos": torch.tensor(global_protos_swa, device=self.device, dtype=torch.float32),
            "ot_target": ot_target
        }

    def aggregate(self):
        if self.args.fed_algorithm == 'FedAvg':
            super().aggregate()
            return

        print('---------------------------')
        print(f'SPIRE aggregate: Starting One-Shot Server-Side Training')

        if self.global_state is None:
            self.global_state = self._one_shot_global_init()
            if self.global_state is None: return

        ctx = self.global_state
        device = self.device

        freeze_b_epochs = int(self.server_epochs * 0.25)

        if self.b_learner is None:
            static_init = ctx.get("ot_target", None)
            # Fallback: if OT init failed, use SWA protos
            if static_init is None:
                static_init = ctx["global_protos"]

            self.b_learner = LearnableBarycenter(
                self.num_classes, self.feature_dim, init_data=static_init
            ).to(self.device)

            self.b_optimizer = torch.optim.Adam(self.b_learner.parameters(), lr=5e-4)
            print("  [SPIRE] Dynamic Barycenter Optimizer Initialized.")

        # Main Server-Side Training Loop
        for epoch in range(self.server_epochs):
            self.round_idx = epoch

            # ======================================================
            # 0. Optimize Dynamic Barycenter (b)
            # ======================================================
            if epoch >= freeze_b_epochs:
                self.b_learner.train()
                self.b_optimizer.zero_grad()
                current_b = self.b_learner()

                loss_b_total = 0.0

                for i, stats in enumerate(ctx["stats_list"]):
                    proto_data = stats["prototypes"]
                    if isinstance(proto_data, torch.Tensor):
                        client_proto = proto_data.clone().detach().to(self.device)
                    else:
                        client_proto = torch.tensor(proto_data, device=self.device)

                    weight = ctx["alphas"][i]

                    # ot Loss (Sinkhorn)
                    l_ot = compute_sinkhorn_loss(current_b, client_proto, epsilon=0.1, niter=3)

                    loss_b_total += weight * l_ot

                loss_b_total.backward()
                self.b_optimizer.step()

            # ======================================================
            # 1. Train Generator (Cosine Warmup)
            # ======================================================
            self.SPIRE_generator.to(self.device)
            self.ema_generator.to(self.device)

            gen_train_steps = getattr(self.args, "gen_train_steps", 50)
            target_w_se = getattr(self.args, "lambda_se", 1e-5)
            target_b_dynamic = self.b_learner().detach()

            # Warmup schedule
            MIN_FACTOR = 0.2
            WARMUP_EPOCHS = max(5, int(self.server_epochs * 0.2))

            if self.round_idx < WARMUP_EPOCHS:
                progress = self.round_idx / float(WARMUP_EPOCHS)
                warmup_factor = MIN_FACTOR + (1 - MIN_FACTOR) * \
                                0.5 * (1 - np.cos(np.pi * progress))
                current_w_se = target_w_se * warmup_factor
            else:
                current_w_se = target_w_se

            if self.gen_mode == "server" or gen_train_steps > 0:
                self.SPIRE_generator.train()
                aligned_cpu = ctx["global_protos"].detach().cpu()
                num_clients = len(ctx["stats_list"])

                for step in range(gen_train_steps):
                    self.gen_optimizer.zero_grad()

                    ACCUM_STEPS = 4
                    for _ in range(ACCUM_STEPS):
                        idx = np.random.randint(num_clients)
                        stats = ctx["stats_list"][idx]
                        stats["prototypes"] = aligned_cpu

                        loss_gen, _, _ = calculate_generator_loss(
                            self.SPIRE_generator, stats, device,
                            lambda_align=getattr(self.args, "lambda_align", 0.0),
                            lambda_se=current_w_se,
                            target_barycenter=target_b_dynamic,
                            batch_size=getattr(self.args, "gen_batch_size", 4096)
                        )
                        loss_gen = loss_gen / ACCUM_STEPS
                        loss_gen.backward()

                    torch.nn.utils.clip_grad_norm_(self.SPIRE_generator.parameters(), max_norm=1.0)
                    self.gen_optimizer.step()

                    # EMA Update
                    ema_decay = 0.995
                    with torch.no_grad():
                        for p_train, p_ema in zip(self.SPIRE_generator.parameters(), self.ema_generator.parameters()):
                            p_ema.data.mul_(ema_decay).add_(p_train.data, alpha=1 - ema_decay)

            # ======================================================
            # 2. Generate Pseudo-Graphs (Generate -> Jitter -> KNN)
            # ======================================================
            self.ema_generator.eval()
            pseudo_graphs = []
            pseudo_alphas = []

            # Annealing schedule
            ANNEAL_END = int(self.server_epochs * 0.75)

            if self.round_idx < ANNEAL_END:
                progress = self.round_idx / ANNEAL_END
                jitter_std = 0.5 * getattr(self.args, "max_jitter", 0.08) * (1 + np.cos(np.pi * progress))
            else:
                jitter_std = 0.0

            base_protos = ctx["global_protos"].detach().cpu()
            k_knn = getattr(self.args, "gen_knn", 5)

            for i, stats in enumerate(ctx["stats_list"]):
                if "num_nodes" not in stats:
                    stats["num_nodes"] = int(stats.get("num_nodes", 20))

                local_n = ctx["adaptive_ns"][i]
                stats["n_samples"] = local_n

                stats["prototypes"] = base_protos

                with torch.no_grad():
                    Gk = self.ema_generator.generate(stats)

                # Apply jitter
                if jitter_std > 1e-6:
                    noise = torch.randn_like(Gk.x) * jitter_std
                    Gk.x = Gk.x + noise

                Gk.edge_index = knn_graph(Gk.x, k=k_knn)

                Gk = Gk.to(device)
                pseudo_graphs.append(Gk)
                pseudo_alphas.append(ctx["alphas"][i])

            # ======================================================
            # 3. Fuse & Pseudo Label
            # ======================================================
            pseudo_alphas = np.array(pseudo_alphas, dtype=np.float32)
            pseudo_alphas /= pseudo_alphas.sum() + 1e-12

            G_global = fuse_graphs(pseudo_graphs, pseudo_alphas, device=device)

            # Detach to stop gradients
            G_global.x = G_global.x.detach()
            if hasattr(G_global, 'edge_index'):
                G_global.edge_index = G_global.edge_index.detach()

            # Pseudo labeling via prototypes
            P_labeling = ctx["global_protos"].detach()
            sims = torch.matmul(F.normalize(G_global.x, p=2, dim=1), F.normalize(P_labeling, p=2, dim=1).t())
            G_global.y = sims.argmax(dim=1)

            del pseudo_graphs
            torch.cuda.empty_cache()

            # ======================================================
            # 4. Train Global Model (GNN)
            # ======================================================
            # Offload generator to CPU to save VRAM
            self.SPIRE_generator.to('cpu')
            self.ema_generator.to('cpu')
            torch.cuda.empty_cache()

            self.model.train()
            self.model.to(device)

            # Determine training strategy
            is_large_graph = G_global.x.size(0) > 500000

            if not is_large_graph:
                # Full-batch training
                g_gpu = G_global.to(device)
                for t in range(self.T_G):
                    self.optimizer.zero_grad()
                    _, out = self.model(g_gpu)
                    logits = out if not isinstance(out, tuple) else out[-1]
                    loss = F.cross_entropy(logits, g_gpu.y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
            else:
                # Mini-batch training via NeighborLoader
                g_cpu = G_global.cpu()
                BATCH_SIZE = getattr(self.args, 'global_batch_size', 8192)
                train_loader = NeighborLoader(
                    g_cpu, num_neighbors=[10, 10],
                    batch_size=BATCH_SIZE, input_nodes=None,
                    shuffle=True, num_workers=4,
                    pin_memory=True, persistent_workers=True
                )
                for t in range(self.T_G):
                    for batch in train_loader:
                        batch = batch.to(device)
                        self.optimizer.zero_grad()
                        _, out = self.model(batch)
                        logits = out if not isinstance(out, tuple) else out[-1]
                        loss = F.cross_entropy(logits[:batch.batch_size], batch.y[:batch.batch_size])
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()

            self.scheduler.step()

            # Cleanup
            G_global = G_global.cpu()
            torch.cuda.empty_cache()

            # ======================================================
            # 5. Evaluation & Checkpointing
            # ======================================================
            # A. Online Eval
            val_acc, test_acc, test_f1, test_loss = self.global_evaluate()

            print(f"[Epoch {epoch + 1}] Online-- Val: {val_acc:.4f} | Test: {test_acc:.4f}")

            if not hasattr(self, 'best_val_acc'):
                self.best_val_acc = 0.0
                self.test_acc_at_best_val = 0.0
                self.early_stop_counter = 0
                self.early_stop_patience = getattr(self.args, 'patience', 30)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.test_acc_at_best_val = test_acc
                self.early_stop_counter = 0

                self._save_checkpoint(f"{self.args.dataset}_best_online_val_seed{self.args.seed}.pth", self.model)
                print(f"[Epoch {epoch + 1}] New Best Online! Val: {val_acc:.4f} | Test: {test_acc:.4f}")
            else:
                self.early_stop_counter += 1

            # B. EMA Eval (Optional)
            if not hasattr(self, 'ema_state_dict'):
                self.ema_state_dict = copy.deepcopy(self.model.state_dict())
            else:
                beta = 0.95
                current_state = self.model.state_dict()
                for key in current_state:
                    self.ema_state_dict[key] = beta * self.ema_state_dict[key] + (1 - beta) * current_state[key]

            # Evaluate EMA
            backup_state = copy.deepcopy(self.model.state_dict())
            self.model.load_state_dict(self.ema_state_dict)
            ema_val_acc, ema_test_acc, _, _ = self.global_evaluate()

            if not hasattr(self, 'best_ema_val_acc'):
                self.best_ema_val_acc = 0.0
                self.ema_test_at_best_val = 0.0

            if ema_val_acc > self.best_ema_val_acc:
                self.best_ema_val_acc = ema_val_acc
                self.ema_test_at_best_val = ema_test_acc
                self._save_checkpoint(f"{self.args.dataset}_best_ema_val_seed{self.args.seed}.pth", self.model)
                self._save_checkpoint(f"{self.args.dataset}_best_gen_seed{self.args.seed}.pth", self.SPIRE_generator)

            self.model.load_state_dict(backup_state)

            # C. Early Stopping Check
            if self.early_stop_counter >= self.early_stop_patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
                print("SPIRE One-Shot Training Finished.")
                print(f"Final (Online): {test_acc:.4f}")
                print(f"Final (EMA):    {ema_test_acc:.4f}")
                break

            if epoch + 1 == self.server_epochs:
               print("SPIRE One-Shot Training Finished.")
               print(f"Final (Online): {test_acc:.4f}")
               print(f"Final (EMA):    {ema_test_acc:.4f}")

        print(f"Best (Online): {self.test_acc_at_best_val:.4f}")
        print(f"Best (EMA):    {self.ema_test_at_best_val:.4f}")
        self.stop_training = True

    def _save_checkpoint(self, filename, model_obj):
        import os
        root_dir = getattr(self.args, 'log_dir', '.')
        save_dir = os.path.join(root_dir, "checkpoints")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(model_obj.state_dict(), os.path.join(save_dir, filename))


class SPIREClient(BaseClient):
    """
    SPIRE Client:
    - Calculates local statistics (Entropy, Prototypes).
    - No local generator training needed in the default 'server' mode.
    """

    def __init__(self, args, model, data):
        super(SPIREClient, self).__init__(args, model, data)
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        self.feature_dim = self.data.x.shape[-1]
        self.num_classes = args.num_classes

    def get_stats(self):
        """
        Compute and return local statistics.
        Removed: degree_histogram, phi.
        Kept: S_k, prototypes.
        """
        # Determine training mask
        if hasattr(self.data, 'train_mask') and self.data.train_mask is not None:
            train_mask = self.data.train_mask.cpu().numpy()
        else:
            # Fallback for full-graph usage (use with caution)
            print("[Warning] No train_mask found! Using ALL nodes for prototypes (Risk of Leakage).")
            train_mask = np.ones(self.data.x.shape[0], dtype=bool)

        # 1. Prepare Data
        num_nodes = int(self.data.x.shape[0])

        try:
            A_sparse = to_scipy_sparse_matrix(self.data.edge_index, num_nodes=num_nodes)
        except Exception as e:
            # Fallback for older PyG versions if needed
            print(f"Warning: Sparse conversion failed, using dense fallback. Error: {e}")
            A_sparse = to_dense_adj(self.data.edge_index, max_num_nodes=num_nodes).squeeze(0).cpu().numpy()

        feats = self.data.x.cpu().numpy()
        labels = self.data.y.cpu().numpy()

        # 2. Compute Structural Entropy (S_k)
        # Based on full topology (A_np) to capture structural complexity
        S_k = graph_structural_entropy(A_sparse)

        # 3. Compute Prototypes
        # MUST use only training data to prevent leakage
        train_feats = feats[train_mask]
        train_labels = labels[train_mask]
        prototypes = compute_class_prototypes(train_feats, train_labels, self.num_classes)

        return {
            "S_k": float(S_k),
            "prototypes": prototypes,
            "num_nodes": num_nodes
        }