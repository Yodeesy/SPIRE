# utils/ot_align.py
import numpy as np
import ot


def wasserstein_prototype_alignment(client_protos, alphas, num_classes, feature_dim, max_iter=10, reg=0.5):
    """
    Computes the Global Prototype Barycenter using Wasserstein Optimal Transport (WB).

    Aligns client prototypes into a unified global space by minimizing the
    weighted Wasserstein distance (Sinkhorn divergence) to all clients.

    Args:
        client_protos (list[np.ndarray]): List of client prototype matrices.
        alphas (list[float]): Weights for each client (sum=1).
        num_classes (int): Number of classes.
        feature_dim (int): Dimension of features.
        max_iter (int): Max iterations for barycenter update.
        reg (float): Entropic regularization term for Sinkhorn.

    Returns:
        P_global (np.ndarray): The aligned global prototypes (num_classes, feature_dim).
    """
    # 1. Filter invalid data
    valid_data = []
    for w, p in zip(alphas, client_protos):
        if p is not None:
            valid_data.append((w, p))

    if not valid_data:
        return np.zeros((num_classes, feature_dim), dtype=np.float32)

    valid_alphas_raw, valid_protos = zip(*valid_data)

    # Normalize weights
    w_sum = sum(valid_alphas_raw)
    valid_alphas = np.array([w / w_sum for w in valid_alphas_raw])

    # Edge case: Single client
    if len(valid_protos) == 1:
        return valid_protos[0]

    # Initialize P_global with Simple Weighted Average (SWA) for warm start
    P_global = np.zeros((num_classes, feature_dim), dtype=np.float32)
    for w, p in zip(valid_alphas, valid_protos):
        P_global += w * p

    print(f"  [WB-OT] Aligning {len(valid_protos)} clients (iter={max_iter}, reg={reg})...")

    # 2. Iterative Barycenter Optimization (Free-support Barycenter)
    # We use a simplified Block Coordinate Descent approach for clarity and control.

    # Uniform distribution prior for class balance
    dist_p = ot.unif(num_classes)
    dist_q = ot.unif(num_classes)

    for it in range(max_iter):
        P_new = np.zeros_like(P_global)

        for idx, P_k in enumerate(valid_protos):
            w_k = valid_alphas[idx]

            # A. Compute Euclidean Cost Matrix
            M = ot.dist(P_k, P_global, metric='euclidean')

            # B. Solve Entropic Regularized OT (Sinkhorn)
            # min <T, M> - reg * H(T)
            try:
                T_k = ot.sinkhorn(dist_p, dist_q, M, reg)
            except Exception:
                # Fallback to exact OT (EMD) if Sinkhorn fails numerically
                T_k = ot.emd(dist_p, dist_q, M)

            # [Defense] Numerical Stability Check
            if T_k is None or np.isnan(T_k).any():
                T_k = np.eye(num_classes) * (1.0 / num_classes)

            # C. Barycentric Mapping (Project P_k to P_global space)
            # Formula: P_aligned = (T_k / sum(T_k, axis=1)) @ P_global?
            # Actually for free support barycenter update: P_new += w_k * (n * T.T @ P_k)
            P_k_aligned = num_classes * np.matmul(T_k.T, P_k)

            P_new += w_k * P_k_aligned

        # Update global target
        P_global = P_new

    return P_global