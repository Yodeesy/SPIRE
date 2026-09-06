# utils/utils.py
import torch
import numpy as np


def get_entropy_guided_samples(stats_list, alphas, eta=1.0):
    """
    [SPIRE Core] EGVA: Entropy-Guided Volume Allocation

    1. Small Clients (< 500 nodes, e.g., partitioned Cora):
       - High need for Augmentation.
       - Low computational cost.
       -> Grant HIGH budget (e.g., n_max=5) to enrich global diversity.

    2. Large Clients (> 2000 nodes, e.g., partitioned PubMed):
       - Low need for Augmentation (already stable).
       - High computational cost (OOM risk).
       -> Grant LOW budget (e.g., n_max=1) to save memory.
    """
    # Extract node counts
    node_counts = np.array([int(s.get("num_nodes", s.get("n_nodes", 0))) for s in stats_list])

    a_max = max(alphas)
    if a_max > 0:
        norm_alphas = np.array(alphas) / a_max
    else:
        norm_alphas = np.zeros_like(alphas)

    adaptive_n = []

    for i, norm_a in enumerate(norm_alphas):
        n_nodes = node_counts[i]

        # --- Step A: Inverse Thresholding ---
        # Logic: Smaller graphs get MORE budget (Data Augmentation).
        #        Larger graphs get LESS budget (OOM Prevention).

        if n_nodes < 500:
            # Case 1: Small Client (e.g., Cora/Citeseer split)
            # Strategy: Aggressive Augmentation
            # Range: [3, 5] (Minimum 3 copies to ensure diversity)
            local_n_min = 3.0
            local_n_max = 5.0

        elif n_nodes < 2000:
            # Case 2: Medium Client
            # Strategy: Moderate Augmentation
            # Range: [1, 3]
            local_n_min = 1.0
            local_n_max = 3.0

        else:
            # Case 3: Large Client (e.g., PubMed split)
            # Strategy: Conservative / Memory Safe
            # Range: [1, 1] (Just one robust copy is enough)
            local_n_min = 1.0
            local_n_max = 1.0

        # --- Step B: Entropy Guidance ---
        # If min == max, no scaling needed
        if local_n_min == local_n_max:
            adaptive_n.append(int(local_n_max))
            continue

        # Scale based on alpha (quality) within the range [min, max]
        # Higher alpha (better structure) -> Closer to max budget
        raw_n_float = local_n_min + eta * norm_a * (local_n_max - local_n_min)

        # Stochastic Rounding
        lower = int(np.floor(raw_n_float))
        prob = raw_n_float - lower
        increment = np.random.binomial(1, prob)
        final_n = int(lower + increment)

        # Clip just in case
        final_n = min(max(final_n, int(local_n_min)), int(local_n_max))
        adaptive_n.append(final_n)

    return adaptive_n