# backbone/GraphDiffusionGenerator.py
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Standard Sinusoidal Positional Encodings for Time t.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class AdaLN(nn.Module):
    """
    [Core Component] Adaptive Layer Normalization (AdaLN)
    """

    def __init__(self, in_dim, cond_dim):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_dim, elementwise_affine=False, eps=1e-6)
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, in_dim * 2)
        )
        with torch.no_grad():
            self.cond_proj[1].weight.zero_()
            self.cond_proj[1].bias.zero_()

    def forward(self, x, cond):
        scale, shift = self.cond_proj(cond).chunk(2, dim=1)
        return self.layernorm(x) * (1 + scale) + shift


class ResDenoiseBlock(nn.Module):
    """
    [Core Component] Deep Residual Denoising Block
    """

    def __init__(self, in_dim, cond_dim, hidden_dim, time_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, hidden_dim)
        )
        self.norm1 = AdaLN(in_dim, cond_dim)
        self.act1 = nn.GELU()
        self.conv1 = nn.Linear(in_dim, hidden_dim)

        self.norm2 = AdaLN(hidden_dim, cond_dim)
        self.act2 = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Linear(hidden_dim, in_dim)

        self.skip = nn.Identity() if in_dim == in_dim else nn.Linear(in_dim, in_dim)

    def forward(self, x, t_emb, cond):
        h = x
        h = self.norm1(h, cond)
        h = self.act1(h)
        h = self.conv1(h)

        t_feat = self.time_mlp(t_emb)
        h = h + t_feat

        h = self.norm2(h, cond)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return self.skip(x) + h


class GraphDiffusionGenerator(nn.Module):
    """
    [Backbone] Graph Diffusion Generator
    Role: Purely generates Node Features (Manifold) conditioned on Semantic & Structural priors.
    Note: Topology construction (KNN) is delegated to the Server logic (SPIRE.py).
    """

    def __init__(self, device, noise_dim, feature_dim, num_classes,
                 num_timesteps=100,
                 hidden_dim=256,
                 num_layers=3,
                 dropout=0.1,
                 args=None):
        super().__init__()
        self.device = device
        self.args = args
        self.noise_dim = noise_dim
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps

        # Condition: Prototypes (feature_dim) + Entropy (1)
        self.cond_dim = feature_dim + 1

        self.init_proj = nn.Linear(noise_dim, hidden_dim)

        time_dim = hidden_dim
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim // 4),
            nn.Linear(hidden_dim // 4, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.layers = nn.ModuleList([
            ResDenoiseBlock(
                in_dim=hidden_dim,
                cond_dim=self.cond_dim,
                hidden_dim=hidden_dim,
                time_dim=time_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        self.final_norm = AdaLN(hidden_dim, self.cond_dim)
        self.final_proj = nn.Linear(hidden_dim, feature_dim)

    def build_cond_tensor(self, stats, device=None, assigned_labels=None):
        """
        Constructs the condition vector 'c'.
        If assigned_labels is provided, it dictates the shape (Batch Size).
        """
        device = device or self.device

        prototypes = stats.get("prototypes", None)
        S_k = float(stats.get("S_k", 0.0))

        # Determine Batch Size
        if assigned_labels is not None:
            curr_num_nodes = assigned_labels.size(0)
        else:
            # Fallback (mostly for training batching)
            curr_num_nodes = int(stats.get("num_nodes", 20))

        cond_parts = []

        # A. Prototypes (Semantic)
        if prototypes is not None:
            p = np.array(prototypes)
            if assigned_labels is not None:
                # Map class labels to specific prototypes
                p_tensor = torch.tensor(p, dtype=torch.float32, device=device)
                cond_proto = p_tensor[assigned_labels]
            else:
                # Fallback: Repeat mean or zero (Should usually not happen in generation)
                p_mean = torch.tensor(p.mean(axis=0), dtype=torch.float32, device=device)
                cond_proto = p_mean.unsqueeze(0).repeat(curr_num_nodes, 1)
        else:
            cond_proto = torch.zeros((curr_num_nodes, self.feature_dim), dtype=torch.float32, device=device)
        cond_parts.append(cond_proto)

        # B. Structural Entropy (Complexity)
        sk_t = torch.tensor([S_k], dtype=torch.float32, device=device).unsqueeze(0).repeat(curr_num_nodes, 1)
        cond_parts.append(sk_t)

        return torch.cat(cond_parts, dim=1)

    def forward(self, z, cond):
        """
        Denoising: x_T (z) -> x_0
        """
        x = self.init_proj(z)
        batch_size = x.size(0)

        for i in range(self.num_timesteps, 0, -1):
            t = torch.tensor([i] * batch_size, device=self.device).float()
            t_emb = self.time_embed(t)

            for layer in self.layers:
                x = layer(x, t_emb, cond)

        x = self.final_norm(x, cond)
        x = self.final_proj(x)
        return torch.tanh(x)

    def generate(self, stats):
        """
        [Updated for SPIRE Logic]
        1. Read 'n_samples' (Budget per class) from stats.
        2. Generate balanced labels (y_gen).
        3. Generate Features (X_hat).
        4. Return Data(x, y). NO KNN HERE.
        """
        # 1. Determine Generation Budget (Per Class)
        # SPIRE passes 'n_samples' (e.g., 1, 3, 5) via EGBA
        n_per_class = int(stats.get("n_samples", 1))

        # Total nodes to generate
        total_gen_nodes = n_per_class * self.num_classes

        # 2. Generate Balanced Labels
        # [0,0,0, 1,1,1, ..., C,C,C]
        y_gen = torch.arange(self.num_classes, device=self.device).repeat_interleave(n_per_class)

        # 3. Build Conditions & Noise
        cond = self.build_cond_tensor(stats, self.device, assigned_labels=y_gen)
        z = torch.randn((total_gen_nodes, self.noise_dim), device=self.device)

        # 4. Generate Features
        X_hat = self.forward(z, cond)

        # 5. Pack into Data Object
        # Note: edge_index is NOT created here. It will be built by the Server after Jittering.
        data = Data(x=X_hat, y=y_gen)

        # Placeholder edge_index to avoid PyG warnings if checked immediately (Optional)
        # data.edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)

        return data