# datasets/dataset_loader.py
import os
import os.path as osp
import torch
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WikiCS, WikipediaNetwork, WebKB, Actor
from ogb.nodeproppred import PygNodePropPredDataset


def get_random_split_masks(num_nodes, labels, train_prop=0.6, valid_prop=0.2, seed=42):
    """
    Generates deterministic random splits based on a seed.
    Commonly used for datasets without official splits (e.g., Amazon, Coauthor).
    """
    rs = np.random.RandomState(seed)
    perm = torch.as_tensor(rs.permutation(num_nodes))

    train_num = int(num_nodes * train_prop)
    valid_num = int(num_nodes * valid_prop)

    train_indices = perm[:train_num]
    val_indices = perm[train_num: train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    return train_mask, val_mask, test_mask


def load_dataset(train_val_test_split, root_dir, dataset_name, seed=42):
    """
    Unified Data Loader enforcing Academic Standards.
    Prioritizes official splits when available.
    """

    # 1. Validate Dataset Support
    supported_datasets = {
        'cora', 'citeseer', 'pubmed',
        'computers', 'photo',
        'ogbn-arxiv', 'ogbn-products',
        'wikics',
        'coauthor-cs', 'coauthor-physics',
        'chameleon', 'squirrel',
        'texas', 'cornell', 'wisconsin',
        'actor'
    }
    assert dataset_name in supported_datasets, f"Invalid dataset: {dataset_name}"

    if train_val_test_split is None:
        # Default split for random split datasets if not specified
        train_val_test_split = [0.2, 0.4, 0.4]

    print(f"[Dataset] Loading {dataset_name} from {root_dir}...")

    # ==============================================================================
    # TYPE 1: Planetoid (Cora, Citeseer, Pubmed)
    # STANDARD: Public Split
    # ==============================================================================
    if dataset_name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=root_dir, name=dataset_name, split='public', transform=T.NormalizeFeatures())
        data = dataset[0]
        return data

    # ==============================================================================
    # TYPE 2: OGB (ogbn-arxiv, ogbn-products)
    # STANDARD: OGB Official Splits
    # ==============================================================================
    elif dataset_name in ['ogbn-arxiv', 'ogbn-products']:
        dataset = PygNodePropPredDataset(name=dataset_name, root=root_dir, transform=T.ToUndirected())
        data = dataset[0]
        data.y = data.y.squeeze()

        # Use OGB Official Split
        split_idx = dataset.get_idx_split()

        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

        data.train_mask[split_idx['train']] = True
        data.val_mask[split_idx['valid']] = True
        data.test_mask[split_idx['test']] = True

        return data

    # ==============================================================================
    # TYPE 3: WikiCS
    # STANDARD: Official 20-fold Splits (Default to split 0)
    # ==============================================================================
    elif dataset_name == 'wikics':
        dataset = WikiCS(root=osp.join(root_dir, 'wikics'), transform=T.NormalizeFeatures())
        data = dataset[0]

        # WikiCS provides 20 official train/val splits and 1 test split
        # Standard practice: use the first split (split 0) unless doing cross-validation
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
        # data.test_mask is already 1D

        return data

    # ==============================================================================
    # TYPE 4: (WikipediaNetwork, WebKB, Actor)
    # STANDARD: Official 10-fold Splits (Default to split 0)
    # ==============================================================================
    elif dataset_name in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root=root_dir, name=dataset_name, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
        data.test_mask = data.test_mask[:, 0]
        return data

    elif dataset_name in ['texas', 'cornell', 'wisconsin']:
        dataset = WebKB(root=root_dir, name=dataset_name, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
        data.test_mask = data.test_mask[:, 0]
        return data

    elif dataset_name == 'actor':
        dataset = Actor(root=osp.join(root_dir, 'actor'), transform=T.NormalizeFeatures())
        data = dataset[0]
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
        data.test_mask = data.test_mask[:, 0]
        return data

    # ==============================================================================
    # TYPE 4: Amazon
    # STANDARD: Random Split (Deterministic)
    # ==============================================================================
    elif dataset_name in ['computers', 'photo']:
        transform = T.Compose([
            T.NormalizeFeatures(),
            T.ToUndirected(),
            T.AddSelfLoops()
        ])
        dataset = Amazon(root=root_dir, name=dataset_name, transform=transform)
        data = dataset[0]
        if data.y.dim() > 1:
            data.y = data.y.squeeze()

    # ==============================================================================
    # TYPE 5: Coauthor (CS, Physics)
    # STANDARD: Random Split (Deterministic)
    # ==============================================================================
    elif dataset_name in ['coauthor-cs', 'coauthor-physics']:
        # Map input name to PyG class expected name ('CS' or 'Physics')
        name_map = {'coauthor-cs': 'CS', 'coauthor-physics': 'Physics'}
        transform = T.Compose([
            T.NormalizeFeatures(),
            T.ToUndirected(),
            T.AddSelfLoops()
        ])
        dataset = Coauthor(
            root=root_dir,
            name=name_map[dataset_name],
            transform=transform
        )
        data = dataset[0]

    # ==============================================================================
    # Random Split Logic (Shared by Amazon & Coauthor)
    # ==============================================================================
    # If we reached here without returning, it means we need to generate random masks
    data.train_mask, data.val_mask, data.test_mask = get_random_split_masks(
        num_nodes=data.num_nodes,
        labels=data.y,
        train_prop=train_val_test_split[0],
        valid_prop=train_val_test_split[1],
        seed=seed
    )

    return data