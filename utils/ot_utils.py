# utils/ot_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableBarycenter(nn.Module):
    """
    A PyTorch module representing the Dynamic Learnable Barycenter (b).

    It holds the learnable parameters 'b' and can be optimized via Gradient Descent
    to minimize the FGW distance to client prototypes.
    """

    def __init__(self, num_classes, feature_dim, init_data=None):
        super().__init__()
        # b is the learnable matrix [Num_Classes, Feature_Dim]
        if init_data is not None:
            # Warm Start: Initialize with static FGW result
            # detach() is crucial to stop gradients from initialization logic
            self.b = nn.Parameter(init_data.clone().detach())
        else:
            # Cold Start: Random initialization
            self.b = nn.Parameter(torch.randn(num_classes, feature_dim))

    def forward(self):
        """Returns the current barycenter 'b'."""
        return self.b


def compute_sinkhorn_loss(x, y, epsilon=0.1, niter=5):
    """
    Computes the Sinkhorn (Wasserstein) distance between distributions x and y.

    This represents the 'Feature Alignment' term It is used to
    optimize the dynamic barycenter 'b' to match the feature distribution
    of client prototypes.

    Args:
        x: Input distribution [N, D] (e.g., Current Barycenter b).
        y: Target distribution [M, D] (e.g., Client Prototypes P).
        epsilon: Regularization parameter for entropy (default: 0.1).
        niter: Number of Sinkhorn iterations (default: 5).

    Returns:
        torch.Tensor: Scalar Sinkhorn loss.
    """
    # 1. Normalization (Ensures stable Cosine similarity)
    x_norm = F.normalize(x, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)

    # 2. Cost Matrix (Cosine Distance)
    # C = 1 - CosineSimilarity
    C = 1 - torch.mm(x_norm, y_norm.t())

    # 3. Initialize Marginals (Assume Uniform Distribution)
    mu = torch.ones(x.shape[0], device=x.device) / x.shape[0]
    nu = torch.ones(y.shape[0], device=y.device) / y.shape[0]

    # 4. Sinkhorn Iterations (Log-domain for stability)
    u = torch.zeros_like(mu)
    v = torch.zeros_like(nu)

    for _ in range(niter):
        # Update dual variable u
        u_term = (u.unsqueeze(1) + v.unsqueeze(0) - C) / epsilon
        u = epsilon * (torch.log(mu) - torch.logsumexp(u_term, dim=1)) + u

        # Update dual variable v
        v_term = (u.unsqueeze(1) + v.unsqueeze(0) - C).t() / epsilon
        v = epsilon * (torch.log(nu) - torch.logsumexp(v_term, dim=1)) + v

    # 5. Optimal Transport Plan P
    P = torch.exp((u.unsqueeze(1) + v.unsqueeze(0) - C) / epsilon)

    # 6. Final Distance (Sum of P * C)
    loss = torch.sum(P * C)

    return loss