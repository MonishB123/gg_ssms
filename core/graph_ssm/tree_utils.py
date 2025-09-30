import torch
import os

def _build_pair_index_gpu(original_pairs: torch.Tensor):
    """Build edge index lookup on GPU using fully vectorized operations.
    Returns a tensor that can be used for vectorized lookups."""
    # Fully vectorized - no Python loops!
    num_pairs = original_pairs.shape[0]
    max_node = original_pairs.max().item() + 1  # Only one CPU sync, unavoidable
    
    # Create a 2D lookup table [max_node, max_node] -> index
    # Initialize with -1 (invalid)
    lookup = torch.full((max_node, max_node), -1, dtype=torch.long, device=original_pairs.device)
    
    # Create index tensor [0, 1, 2, ..., num_pairs-1]
    indices = torch.arange(num_pairs, dtype=torch.long, device=original_pairs.device)
    
    # Extract u and v columns (stays on GPU)
    u = original_pairs[:, 0].long()
    v = original_pairs[:, 1].long()
    
    # Vectorized assignment for both directions
    lookup[u, v] = indices  # (u,v) -> index
    lookup[v, u] = indices  # (v,u) -> index (bidirectional)
    
    return lookup

def _edge_weights_for_batch_edges_gpu(batch_edges, weights_row, lookup_table):
    """GPU-native edge weight lookup."""
    # Use the lookup table to find indices
    u = batch_edges[:, 0].long()
    v = batch_edges[:, 1].long()
    
    # Get indices from lookup table
    indices = lookup_table[u, v]
    
    # Create weights tensor, using 0.0 for missing edges (index == -1)
    weights = torch.where(
        indices >= 0,
        weights_row[indices.clamp(min=0)],  # clamp to avoid negative indexing
        torch.zeros_like(weights_row[0])
    )
    
    return weights

def _is_connected(edges):
    """Check if a set of edges forms a connected graph using DFS.
    
    Args:
        edges: [E,2] tensor of edges
        
    Returns:
        bool: True if graph is connected
    """
    if edges.numel() == 0:
        return True  # Empty graph is considered connected
        
    # Convert edges to adjacency list format
    nodes = edges.to(torch.int64)
    max_node = nodes.max().item()
    adj_list = [[] for _ in range(max_node + 1)]
    
    # Build adjacency list (bidirectional since undirected graph)
    for u, v in edges.tolist():
        if u == 0 and v == 0:  # Skip padding
            continue
        adj_list[u].append(v)
        adj_list[v].append(u)
    
    # Run DFS
    visited = set()
    def dfs(node):
        visited.add(node)
        for neighbor in adj_list[node]:
            if neighbor not in visited:
                dfs(neighbor)
                
    # Start DFS from first non-zero node we find
    start_node = None
    for u, v in edges.tolist():
        if not (u == 0 and v == 0):
            start_node = u
            break
            
    if start_node is not None:
        dfs(start_node)
        
    # Check if all non-zero nodes are visited
    non_zero_nodes = set()
    for u, v in edges.tolist():
        if not (u == 0 and v == 0):
            non_zero_nodes.add(u)
            non_zero_nodes.add(v)
            
    return len(non_zero_nodes) == len(visited)

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

def save_edge_analysis(tree: torch.Tensor, result: torch.Tensor, original_pairs: torch.Tensor, weights: torch.Tensor, output_dir: str = "core/graph_ssm/tree_utils_tests/outputs"):
    """Save edge analysis comparing original and padded trees to a markdown file.
    
    Args:
        tree: [B, E_mst, 2] Original MST edges
        result: [B, E_kept_max, 2] Pruned & padded trees
        original_pairs: [E_orig, 2] Original edge pairs
        weights: [B, E_orig] Edge weights
        output_dir: Output directory for the markdown file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
        
    # Process all batches
    if result.shape[0] > 0:
        # Save to markdown file
        analysis_file = os.path.join(output_dir, "edge_analysis_padded.md")
        with open(analysis_file, 'w') as f:
            # Process each batch
            for batch_idx in range(result.shape[0]):
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
                
                # Write batch header
                f.write(f"\n## Batch {batch_idx}\n\n")
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

def prune_tree_by_weight(
    tree: torch.Tensor,
    original_pairs: torch.Tensor,
    weights: torch.Tensor,
    threshold: float,
    debug: bool = False,
    check_connectivity: bool = False
):
    """Prune edges with weight >= threshold, leaf-only (single pass).
       Only prune edges that are leaves AND >= threshold.
    Args:
        tree: [B, E_mst, 2] MST edges (undirected)
        original_pairs: [E_orig, 2]
        weights: [B, E_orig]
        threshold: float
        debug: If True, print debug info and save analysis (slow!)
        check_connectivity: If True, check connectivity (slow!)
    Returns:
        [B, E_kept_max, 2] pruned & padded trees
    """
    assert tree.dim() == 3 and tree.size(-1) == 2
    assert weights.dim() == 2
    device = tree.device
    
    # Cache lookup table - only build once per unique original_pairs tensor
    if not hasattr(prune_tree_by_weight, '_lookup_cache'):
        prune_tree_by_weight._lookup_cache = {}
    
    # Use id of original_pairs as cache key
    cache_key = id(original_pairs)
    if cache_key not in prune_tree_by_weight._lookup_cache:
        lookup_table = _build_pair_index_gpu(original_pairs)
        prune_tree_by_weight._lookup_cache[cache_key] = lookup_table
    else:
        lookup_table = prune_tree_by_weight._lookup_cache[cache_key]

    # Process all batches in parallel where possible
    pruned_trees = []
    for b in range(tree.shape[0]):
        edges_b = tree[b]
        
        # GPU-native weight lookup
        ew = _edge_weights_for_batch_edges_gpu(edges_b, weights[b], lookup_table)

        # GPU-native leaf mask
        leaf_mask = _leaf_mask_for_edges(edges_b)
        
        # keep if NOT a leaf OR weight < threshold
        keep_mask = (~leaf_mask) | (ew < threshold)
        
        pruned = edges_b[keep_mask]
        
        # Safety: if all edges were pruned, keep the edge with minimum weight
        # This prevents creating a completely disconnected graph
        if pruned.shape[0] == 0:
            min_idx = ew.argmin()
            pruned = edges_b[min_idx:min_idx+1]

        pruned_trees.append(pruned)

    # Pad to same length across batch (stay on GPU)
    # Pad by duplicating the last valid edge instead of using [0,0]
    # This avoids creating self-loops that cause BFS to hang
    # Note: We guarantee at least 1 edge per tree above, so t[-1:] is always safe
    max_edges = max((t.shape[0] for t in pruned_trees), default=1)
    padded = []
    for t in pruned_trees:
        if t.shape[0] < max_edges:
            # Duplicate the last valid edge for padding
            # Safe because we guarantee >= 1 edge per tree
            last_edge = t[-1:]
            num_pads = max_edges - t.shape[0]
            pad = last_edge.repeat(num_pads, 1)
            t = torch.cat([t, pad], dim=0)
        padded.append(t)

    result = torch.stack(padded) if padded else torch.empty_like(tree)
    
    # Only do expensive operations if debug mode is on
    if debug:
        save_edge_analysis(tree, result, original_pairs, weights)
    
    # Only check connectivity if requested (expensive!)
    if check_connectivity:
        all_connected = True
        for b in range(result.shape[0]):
            if not _is_connected(result[b]):
                print(f"\nWARNING: Tree in batch {b} is disconnected after pruning!")
                print("This may indicate the pruning threshold is too aggressive.")
                all_connected = False
        
        if all_connected:
            print("\nAll trees remain connected after pruning.")
    
    return result
