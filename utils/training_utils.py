# utils/training_utils.py
import copy
import torch
import torch.nn.functional as F


def average_state_dicts(state_dicts):
    """
    Aggregates a list of state_dicts by averaging their parameters.
    """
    if len(state_dicts) == 0:
        return None

    avg_state = copy.deepcopy(state_dicts[0])
    keys = list(avg_state.keys())

    for k in keys:
        if avg_state[k].dtype.is_floating_point:
            avg_state[k] = avg_state[k].float() * 0.0
        else:
            continue

    num_states = len(state_dicts)
    for sd in state_dicts:
        for k in keys:
            if avg_state[k].dtype.is_floating_point:
                avg_state[k] += sd[k].float() / num_states

    return avg_state


def differentiable_structure_entropy(x):
    """
    Calculates a differentiable approximation of Structural Entropy.

    Logic:
    1. Construct a 'soft' adjacency matrix based on feature similarity.
    2. Compute the transition matrix P and stationary distribution pi.
    3. Calculate the entropy of the random walk on this soft graph.

    Args:
        x: Generated node features [Batch, Dim]
    Returns:
        H: Scalar structural entropy
    """
    # 1. Soft Adjacency via Cosine Similarity
    # Normalize features to [-1, 1] range
    x = x + 1e-6
    x_norm = F.normalize(x, p=2, dim=1)
    # Cosine similarity matrix. Use ReLU to ensure non-negative weights (graph assumption)
    adj = F.relu(torch.mm(x_norm, x_norm.t()))

    # 2. Transition Matrix P
    # Row normalization: P_ij = A_ij / d_i
    # Add epsilon to prevent division by zero
    row_sum = adj.sum(dim=1, keepdim=True).clamp(min=1e-6)
    P = adj / row_sum

    # 3. Stationary Distribution pi
    # For symmetric adjacency (undirected), pi_i = d_i / vol
    vol = row_sum.sum()
    pi = row_sum / (vol + 1e-6)

    # 4. Structural Entropy (2D)
    # H = - sum(pi_i * sum(P_ij * log(P_ij)))
    # We use natural log for gradient stability
    # P * log(P) -> 0 if P -> 0
    term = P * torch.log(P + 1e-6)
    entropy_per_node = -torch.sum(term, dim=1, keepdim=True)

    # Average over the stationary distribution
    H = torch.sum(pi * entropy_per_node)
    H = torch.tanh(H)
    if torch.isnan(H) or torch.isinf(H):
        return torch.tensor(0.0, device=x.device, requires_grad=True)

    return H


def calculate_generator_loss(generator, stats, device,
                             lambda_align=0.0, lambda_se=0.0,
                             target_barycenter=None,
                             batch_size=4096):
    """
    Computes the loss for training the Graph Diffusion Generator.
    Uses Mini-batching to prevent OOM on large graphs.
    """

    # 1. Determine safe batch size
    full_num_nodes = int(stats.get("num_nodes", 20))
    curr_batch_size = min(full_num_nodes, batch_size)

    # Update stats temporarily to reflect batch size
    batch_stats = stats.copy()
    batch_stats["num_nodes"] = curr_batch_size

    batch_stats["S_k"] = 0.0

    # 2. Randomly sample labels for this batch
    y_gen = torch.randint(0, generator.num_classes, (curr_batch_size,), device=device)

    # 3. Build conditions
    cond = generator.build_cond_tensor(batch_stats, device, assigned_labels=y_gen)

    # 4. Sample noise
    z = torch.randn((curr_batch_size, generator.noise_dim), device=device)

    # 5. Forward pass -> Generate Features X_hat
    X_hat = generator.forward(z, cond)

    # Initialize Loss
    loss = torch.tensor(0.0, device=device)
    log_dict = {}

    # -----------------------------------------------------------
    # 6. Barycenter Matching Loss (L_bary)
    # -----------------------------------------------------------
    if lambda_align > 0 and target_barycenter is not None:
        if not isinstance(target_barycenter, torch.Tensor):
            target_ot = torch.tensor(target_barycenter, device=device, dtype=torch.float32)
        else:
            target_ot = target_barycenter.to(device)

        y_one_hot = F.one_hot(y_gen, generator.num_classes).float()
        counts = y_one_hot.sum(dim=0).unsqueeze(1)
        sums = torch.matmul(y_one_hot.t(), X_hat)
        means = sums / counts.clamp(min=1.0)

        mask_non_empty = (counts > 0)
        batch_centers = torch.where(mask_non_empty, means, target_ot)
        loss_ot_val = F.mse_loss(batch_centers, target_ot)
        loss = loss + lambda_align * loss_ot_val
        log_dict['bary'] = loss_ot_val.item()

    # -----------------------------------------------------------
    # 7. Structural Entropy Loss (L_se)
    # -----------------------------------------------------------
    if lambda_se > 0:
        current_S = differentiable_structure_entropy(X_hat)

        loss_se = current_S

        if loss_se.dim() > 0:
            loss_se = loss_se.mean()

        loss = loss + lambda_se * loss_se
        log_dict['se'] = loss_se.item()

    return loss, None, log_dict