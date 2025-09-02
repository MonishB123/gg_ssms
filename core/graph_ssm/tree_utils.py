# Includes functions for token reduction by pruning/merging nodes in the MST
import torch

def prune_tree_by_weight(tree, weights, threshold):
    """Prunes edges from the tree whose weights are below the threshold.
    
    Args:
        tree: Tensor of shape [batch_size, num_edges, 2] containing edge indices
        weights: Tensor of shape [batch_size, num_edges] containing edge weights
        threshold: Float value for pruning threshold
        
    Returns:
        Pruned tree tensor of shape [batch_size, num_remaining_edges, 2]
    """
    # Create mask for edges to keep
    mask = weights > threshold  # [batch_size, num_edges]
    
    # Apply mask to tree tensor for each batch
    pruned_trees = []
    for batch_idx in range(tree.shape[0]):
        batch_mask = mask[batch_idx]
        pruned_tree = tree[batch_idx][batch_mask]  # Select only edges above threshold
        pruned_trees.append(pruned_tree)
    
    # Pad to same length if needed (since different batches might have different numbers of edges after pruning)
    max_edges = max(t.shape[0] for t in pruned_trees)
    padded_trees = []
    for pruned_tree in pruned_trees:
        if pruned_tree.shape[0] < max_edges:
            padding = torch.zeros((max_edges - pruned_tree.shape[0], 2), 
                                dtype=pruned_tree.dtype, 
                                device=pruned_tree.device)
            padded_tree = torch.cat([pruned_tree, padding], dim=0)
        else:
            padded_tree = pruned_tree
        padded_trees.append(padded_tree)
    
    return torch.stack(padded_trees)
