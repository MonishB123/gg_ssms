"""
Pruning utilities for tree-based structures.

This module provides functions for identifying and pruning leaf nodes in tree structures,
useful for Graph SSM models that build and traverse tree structures.
"""

import torch


def find_leaf_nodes(tree):
    """
    Find leaf nodes in the tree structure.
    
    Args:
        tree: Tensor [batch, num_edges, 2] where each edge is [source, target]
    
    Returns:
        leaf_nodes_batch: List of leaf node indices for each batch
        leaf_edges_batch: List of edge indices connecting to leaf nodes
    """
    batch_size, num_edges, _ = tree.shape
    num_nodes = num_edges + 1  # Tree property: N nodes, N-1 edges
    
    leaf_nodes_batch = []
    leaf_edges_batch = []
    
    for b in range(batch_size):
        # Count degree of each node
        degree = torch.zeros(num_nodes, dtype=torch.int32, device=tree.device)
        
        for edge_idx in range(num_edges):
            src = tree[b, edge_idx, 0].item()
            dst = tree[b, edge_idx, 1].item()
            degree[src] += 1
            degree[dst] += 1
        
        # Leaf nodes have degree 1
        leaf_nodes = torch.where(degree == 1)[0].tolist()
        
        # Find edges connected to leaf nodes
        leaf_edges = []
        for edge_idx in range(num_edges):
            src = tree[b, edge_idx, 0].item()
            dst = tree[b, edge_idx, 1].item()
            if src in leaf_nodes or dst in leaf_nodes:
                leaf_edges.append(edge_idx)
        
        leaf_nodes_batch.append(leaf_nodes)
        leaf_edges_batch.append(leaf_edges)
    
    return leaf_nodes_batch, leaf_edges_batch


def prune_leaf_nodes(tree, edge_weights=None, num_leaves_to_prune=1, verbose=False):
    """
    Create a mask for edges to prune based on leaf nodes.
    
    Prunes N leaf nodes with the HIGHEST edge weights (strongest connections).
    
    Args:
        tree: Tensor [batch, num_edges, 2]
        edge_weights: Optional tensor [batch, num_edges] of edge weights from MST
        num_leaves_to_prune: Number of leaf nodes to remove
        verbose: Whether to print debug information
    
    Returns:
        edge_mask: Boolean tensor [batch, num_edges] where False = pruned, True = keep
        num_removed: Number of edges marked for removal per batch
    """
    batch_size, num_edges, _ = tree.shape
    edge_mask = torch.ones(batch_size, num_edges, dtype=torch.bool, device=tree.device)
    num_removed_per_batch = []
    
    for b in range(batch_size):
        # Find all leaf nodes
        leaf_nodes, leaf_edges = find_leaf_nodes(tree[b:b+1])
        
        if len(leaf_nodes[0]) == 0:
            num_removed_per_batch.append(0)
            continue
        
        # Collect (leaf_node, edge_idx, weight) for all leaves
        leaf_info = []
        for leaf_node in leaf_nodes[0]:
            # Find the edge connected to this leaf
            for edge_idx in range(num_edges):
                src = tree[b, edge_idx, 0].item()
                dst = tree[b, edge_idx, 1].item()
                
                if src == leaf_node or dst == leaf_node:
                    if edge_weights is not None:
                        weight = edge_weights[b, edge_idx].item()
                    else:
                        weight = 0.0  # Default weight if not provided
                    
                    leaf_info.append({
                        'leaf_node': leaf_node,
                        'edge_idx': edge_idx,
                        'weight': weight,
                        'edge': (src, dst)
                    })
                    break
        
        # Sort leaves by weight (descending) - highest weights first
        leaf_info.sort(key=lambda x: x['weight'], reverse=True)
        
        # Prune the N leaves with highest weights
        num_to_prune = min(num_leaves_to_prune, len(leaf_info))
        
        if verbose and b == 0:  # Print debug info for batch 0
            print(f"   Batch 0: Found {len(leaf_info)} leaf nodes")
            print(f"   Pruning {num_to_prune} leaves with HIGHEST weights:")
        
        for i in range(num_to_prune):
            info = leaf_info[i]
            edge_mask[b, info['edge_idx']] = False
            
            if verbose and b == 0:  # Print first few for batch 0
                src, dst = info['edge']
                print(f"   ✗ Pruning edge {info['edge_idx']}: {src}→{dst} (leaf={info['leaf_node']}, weight={info['weight']:.4f})")
        
        if verbose and b == 0 and num_to_prune < len(leaf_info):
            print(f"   Keeping {len(leaf_info) - num_to_prune} leaves with lower weights")
            # Show a few kept leaves
            for i in range(num_to_prune, min(num_to_prune + 3, len(leaf_info))):
                info = leaf_info[i]
                src, dst = info['edge']
                print(f"   ✓ Keeping edge {info['edge_idx']}: {src}→{dst} (leaf={info['leaf_node']}, weight={info['weight']:.4f})")
        
        num_removed_per_batch.append(num_to_prune)
    
    return edge_mask, num_removed_per_batch

