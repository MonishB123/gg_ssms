import torch

def _build_pair_index(original_pairs: torch.Tensor):
    """Bidirectional (u,v)->col index map for O(1) lookups."""
    idx_map = {}
    for i, (u, v) in enumerate(original_pairs.tolist()):
        idx_map[(u, v)] = i
        idx_map[(v, u)] = i
    return idx_map

def _edge_weights_for_batch_edges(batch_edges, weights_row, pair_index, device):
    cols = [pair_index.get(tuple(e.tolist()), None) for e in batch_edges]
    # fill missing with 0.0
    return torch.stack([
        (weights_row[c] if c is not None else torch.tensor(0.0, device=device, dtype=weights_row.dtype))
        for c in cols
    ])

def _leaf_mask_for_edges(batch_edges):
    """
    Given edges [E,2] (int), return boolean mask [E]
    where True means the edge is a 'leaf edge' (at least one endpoint has degree 1).
    """
    if batch_edges.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool, device=batch_edges.device)

    nodes = batch_edges.to(torch.int64)
    max_node = nodes.max().item()
    deg = torch.bincount(nodes.view(-1), minlength=max_node + 1)
    u, v = nodes[:, 0], nodes[:, 1]
    return (deg[u] == 1) | (deg[v] == 1)

def prune_tree_by_weight(
    tree: torch.Tensor,
    original_pairs: torch.Tensor,
    weights: torch.Tensor,
    threshold: float,
    leaf_only: bool = False,
):
    """Prune edges with weight <= threshold.
       If leaf_only=True, only prune edges that are leaves AND <= threshold (single pass).
    Args:
        tree: [B, E_mst, 2] MST edges (undirected)
        original_pairs: [E_orig, 2]
        weights: [B, E_orig]
        threshold: float
        leaf_only: bool – single-pass leaf-only pruning
    Returns:
        [B, E_kept_max, 2] pruned & padded trees
    """
    assert tree.dim() == 3 and tree.size(-1) == 2
    assert weights.dim() == 2
    device = tree.device
    pair_index = _build_pair_index(original_pairs)

    pruned_trees = []
    for b in range(tree.shape[0]):
        edges_b = tree[b]
        ew = _edge_weights_for_batch_edges(edges_b, weights[b], pair_index, device)

        if leaf_only:
            leaf_mask = _leaf_mask_for_edges(edges_b)
            # keep if NOT a leaf OR weight > threshold
            keep_mask = (~leaf_mask) | (ew > threshold)
        else:
            # prune any edge with weight <= threshold
            keep_mask = (ew > threshold)

        pruned_trees.append(edges_b[keep_mask])

    # pad to same length across batch
    max_edges = max((t.shape[0] for t in pruned_trees), default=0)
    padded = []
    for t in pruned_trees:
        if t.shape[0] < max_edges:
            pad = torch.zeros((max_edges - t.shape[0], 2), dtype=t.dtype, device=t.device)
            t = torch.cat([t, pad], dim=0)
        padded.append(t)

    return torch.stack(padded) if padded else torch.empty_like(tree)

# ---- Test harness ----

def create_test_case():
    pairs = torch.tensor([
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
        [5, 6], [6, 7], [7, 8], [8, 9], [8, 10],
        [8, 11], [9, 10], [9, 11], [10, 11]
    ], dtype=torch.int32)

    weights = torch.tensor([[
        0.9635, 0.6404, 0.8880, 0.9658, 0.8253,
        0.7223, 0.6875, 0.8048, 0.6865, 0.8971,
        0.7102, 0.6896, 0.6908, 0.6234
    ]])

    tree = torch.tensor([[
        [0, 1], [1, 2], [2, 3], [4, 5], [5, 6],
        [6, 7], [8, 9], [10, 11], [3, 4], [7, 8],
        [9, 10]
    ]], dtype=torch.int32)

    return tree, pairs, weights

if __name__ == "__main__":
    tree, pairs, weights = create_test_case()
    threshold = 0.7

    print("Original tree:")
    print(tree.squeeze(0))

    print("\nPruned tree (any edge <= threshold):")
    print(prune_tree_by_weight(tree, pairs, weights, threshold, leaf_only=False).squeeze(0))

    print("\nPruned tree (leaf-only, single pass):")
    print(prune_tree_by_weight(tree, pairs, weights, threshold, leaf_only=True).squeeze(0))
