import torch
import os

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
    threshold: float
):
    """Prune edges with weight >= threshold, leaf-only (single pass).
       Only prune edges that are leaves AND >= threshold.
    Args:
        tree: [B, E_mst, 2] MST edges (undirected)
        original_pairs: [E_orig, 2]
        weights: [B, E_orig]
        threshold: float
    Returns:
        [B, E_kept_max, 2] pruned & padded trees
    """

    # Print the tree
    print("Tree before pruning:")
    print(tree)
    # Print the weights
    print("Weights before pruning:")
    print(weights)
    assert tree.dim() == 3 and tree.size(-1) == 2
    assert weights.dim() == 2
    device = tree.device
    pair_index = _build_pair_index(original_pairs)

    pruned_trees = []
    for b in range(tree.shape[0]):
        edges_b = tree[b]
        ew = _edge_weights_for_batch_edges(edges_b, weights[b], pair_index, device)

        leaf_mask = _leaf_mask_for_edges(edges_b)
        # keep if NOT a leaf OR weight < threshold
        keep_mask = (~leaf_mask) | (ew < threshold)

        pruned_trees.append(edges_b[keep_mask])

    # pad to same length across batch
    max_edges = max((t.shape[0] for t in pruned_trees), default=0)
    padded = []
    for t in pruned_trees:
        if t.shape[0] < max_edges:
            pad = torch.zeros((max_edges - t.shape[0], 2), dtype=t.dtype, device=t.device)
            t = torch.cat([t, pad], dim=0)
        padded.append(t)

    result = torch.stack(padded) if padded else torch.empty_like(tree)
    
    # Print padded results after padding occurs
    # Hardcoded output directory
    output_dir = "core/graph_ssm/tree_utils_tests/outputs"
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
        
    # Process only the first batch for simplicity
    if result.shape[0] > 0:
        batch_idx = 0
        padded_tree_b = result[batch_idx]
        weights_b = weights[batch_idx]
        
        # Create a mapping from edge pairs to weights
        pair_to_weight = {}
        for i, (u, v) in enumerate(original_pairs.tolist()):
            pair_to_weight[(u, v)] = weights_b[i].item()
            pair_to_weight[(v, u)] = weights_b[i].item()  # bidirectional
        
        # Get original MST edges and weights (before padding)
        original_edges_list = [e for e in tree[batch_idx].tolist() if not (e[0] == 0 and e[1] == 0)]
        original_weights_list = [round(pair_to_weight.get((u, v), 0.0), 4) for u, v in original_edges_list]
        
        # Get padded edges and weights (after padding)
        padded_edges_list = padded_tree_b.tolist()
        padded_weights_list = [round(pair_to_weight.get((u, v), 0.0), 4) if not (u == 0 and v == 0) else "" for u, v in padded_edges_list]
        
        # Save to markdown file
        analysis_file = os.path.join(output_dir, "edge_analysis_padded.md")
        with open(analysis_file, 'w') as f:
            f.write("| Original Edges | Original Weights | Padded Edges | Padded Weights |\n")
            f.write("|----------------|------------------|--------------|----------------|\n")
            
            # Find the maximum length to pad shorter lists
            max_len = max(len(original_edges_list), len(padded_edges_list))
            
            # Pad shorter list with empty strings
            original_edges_padded = original_edges_list + [""] * (max_len - len(original_edges_list))
            original_weights_padded = original_weights_list + [""] * (max_len - len(original_weights_list))
            padded_edges_padded = padded_edges_list + [""] * (max_len - len(padded_edges_list))
            padded_weights_padded = padded_weights_list + [""] * (max_len - len(padded_weights_list))
            
            # Write each row
            for i in range(max_len):
                f.write(f"| {original_edges_padded[i]} | {original_weights_padded[i]} | {padded_edges_padded[i]} | {padded_weights_padded[i]} |\n")
        
        print(f"\nSaved padded edge analysis to: {analysis_file}")
    

    # Print the result
    print("Edges after pruning:")
    print(result)
    print("Weights after pruning:")
    print(weights)
    
    return result
