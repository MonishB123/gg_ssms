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
):
    """Prune edges with weight <= threshold, leaf-only (single pass).
       Only prune edges that are leaves AND <= threshold.
    Args:
        tree: [B, E_mst, 2] MST edges (undirected)
        original_pairs: [E_orig, 2]
        weights: [B, E_orig]
        threshold: float
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

        leaf_mask = _leaf_mask_for_edges(edges_b)
        # keep if NOT a leaf OR weight > threshold
        keep_mask = (~leaf_mask) | (ew > threshold)

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
