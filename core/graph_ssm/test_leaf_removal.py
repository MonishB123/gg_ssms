"""
Test script to understand what breaks when removing leaf nodes and edges from the MST.

This will help us understand the downstream effects before implementing full pruning.
"""

import torch


def find_leaf_nodes(tree):
    """
    Find leaf nodes in the tree structure.
    
    Args:
        tree: Tensor [batch, num_edges, 2] where each edge is [source, target]
    
    Returns:
        leaf_nodes: List of leaf node indices for each batch
        leaf_edges: List of edge indices connecting to leaf nodes
    """
    batch_size, num_edges, _ = tree.shape
    num_nodes = num_edges + 1  # Tree property: N nodes, N-1 edges
    
    leaf_nodes_batch = []
    leaf_edges_batch = []
    
    for b in range(batch_size):
        # Count degree of each node
        degree = torch.zeros(num_nodes, dtype=torch.int32)
        
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


def remove_leaf_node(tree, batch_idx, leaf_node_idx):
    """
    Remove a specific leaf node and its connecting edge from the tree.
    
    Args:
        tree: Tensor [batch, num_edges, 2]
        batch_idx: Which batch to modify
        leaf_node_idx: Index of the leaf node to remove
    
    Returns:
        modified_tree: New tree with the edge removed (marked as -1)
        removed_edge_idx: Index of the removed edge
    """
    modified_tree = tree.clone()
    num_edges = tree.shape[1]
    removed_edge_idx = None
    
    # Find the edge connected to this leaf node
    for edge_idx in range(num_edges):
        src = tree[batch_idx, edge_idx, 0].item()
        dst = tree[batch_idx, edge_idx, 1].item()
        
        if src == leaf_node_idx or dst == leaf_node_idx:
            # Mark this edge as removed
            modified_tree[batch_idx, edge_idx, 0] = -1
            modified_tree[batch_idx, edge_idx, 1] = -1
            removed_edge_idx = edge_idx
            break
    
    return modified_tree, removed_edge_idx


def print_tree_structure(tree, batch_idx=0):
    """Print the tree structure for visualization."""
    num_edges = tree.shape[1]
    print(f"\nTree structure (batch {batch_idx}):")
    for edge_idx in range(num_edges):
        src = tree[batch_idx, edge_idx, 0].item()
        dst = tree[batch_idx, edge_idx, 1].item()
        if src != -1 and dst != -1:
            print(f"  Edge {edge_idx}: {src} → {dst}")
        else:
            print(f"  Edge {edge_idx}: REMOVED")


def test_leaf_removal():
    """
    Main test function to see what breaks when we remove leaf nodes.
    """
    print("=" * 70)
    print("Test: Removing Leaf Nodes from Tree")
    print("=" * 70)
    
    # Create a simple sequential tree
    batch_size = 2
    seq_len = 10
    d_model = 16
    
    # Create a simple tree: 0→1→2→3→4→5→6→7→8→9 (linear chain)
    num_edges = seq_len - 1
    tree = torch.zeros(batch_size, num_edges, 2, dtype=torch.int32)
    for b in range(batch_size):
        for i in range(num_edges):
            tree[b, i, 0] = i
            tree[b, i, 1] = i + 1
    
    print("\n1. Original Tree Structure")
    print("-" * 70)
    print_tree_structure(tree, batch_idx=0)
    
    # Find leaf nodes
    leaf_nodes, leaf_edges = find_leaf_nodes(tree)
    print(f"\nLeaf nodes (batch 0): {leaf_nodes[0]}")
    print(f"Edges connecting to leaves (batch 0): {leaf_edges[0]}")
    
    # Remove one leaf node
    print("\n2. Removing Leaf Node")
    print("-" * 70)
    leaf_to_remove = leaf_nodes[0][0]  # Remove first leaf from batch 0
    print(f"Removing leaf node: {leaf_to_remove}")
    
    modified_tree, removed_edge = remove_leaf_node(tree, batch_idx=0, leaf_node_idx=leaf_to_remove)
    print(f"Removed edge index: {removed_edge}")
    print_tree_structure(modified_tree, batch_idx=0)
    
    # Count remaining edges
    valid_edges = (modified_tree[0, :, 0] != -1).sum().item()
    print(f"\nOriginal edges: {num_edges}")
    print(f"Remaining edges: {valid_edges}")
    print(f"Removed edges: {num_edges - valid_edges}")
    
    # Analyze what would need to change downstream
    print("\n3. Impact Analysis")
    print("-" * 70)
    print("If we pass this modified tree to downstream operations:")
    print(f"  - Original tree has {num_edges} edges connecting {num_edges + 1} nodes")
    print(f"  - Modified tree has {valid_edges} valid edges")
    print(f"  - {num_edges - valid_edges} edges are marked as -1 (invalid)")
    print(f"\nPotential issues:")
    print(f"  1. BFS traversal will encounter -1 node indices")
    print(f"  2. Tree refinement operations may try to access invalid nodes")
    print(f"  3. Batch processing assumes all batches have same number of edges")
    print(f"  4. Feature gathering (torch.index_select) may fail with -1 indices")
    
    # Test feature indexing with -1
    print("\n4. Testing Feature Indexing with -1 Indices")
    print("-" * 70)
    dummy_features = torch.randn(batch_size, d_model, seq_len)
    print(f"Dummy features shape: {dummy_features.shape}")
    
    print("\nTest 1: Index with valid node (node 5):")
    try:
        result = torch.index_select(dummy_features, 2, torch.tensor([5]))
        print(f"  ✓ Success! Result shape: {result.shape}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    print("\nTest 2: Index with -1 (invalid node):")
    try:
        result = torch.index_select(dummy_features, 2, torch.tensor([-1]))
        print(f"  ✓ Unexpectedly succeeded! Result shape: {result.shape}")
        print(f"  (PyTorch treats -1 as last index)")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    print("\n5. Implications for Pruning")
    print("-" * 70)
    print("Key findings:")
    print("  1. Leaf nodes in a linear chain are the endpoints (nodes 0 and 9)")
    print("  2. Removing a leaf node reduces the tree size")
    print("  3. Marking edges as -1 preserves tensor shape but creates issues:")
    print("     - -1 in PyTorch indexing means 'last element' (not 'invalid')")
    print("     - CUDA kernels (BFS, tree_scan) won't recognize -1 as invalid")
    print("  4. For pruning to work, we have three options:")
    print("     a) Reindex nodes and compact the tree (complex, breaks batching)")
    print("     b) Modify CUDA kernels to skip edges with -1 indices")
    print("     c) Use a separate edge mask and filter before passing to CUDA")
    
    return tree, modified_tree, leaf_nodes, leaf_edges


def test_with_more_complex_tree():
    """
    Test with a tree that has more branches (not just a linear chain).
    """
    print("\n\n" + "=" * 70)
    print("Test: More Complex Tree with Branches")
    print("=" * 70)
    
    batch_size = 1
    
    # Create a tree with branches:
    #       0
    #      / \
    #     1   2
    #    / \   \
    #   3   4   5
    #  /
    # 6
    
    edges = [
        [0, 1],
        [0, 2],
        [1, 3],
        [1, 4],
        [2, 5],
        [3, 6],
    ]
    
    num_edges = len(edges)
    num_nodes = num_edges + 1  # 7 nodes
    tree = torch.tensor([[edges]], dtype=torch.int32).squeeze(1)
    
    print("\nTree structure:")
    print("       0")
    print("      / \\")
    print("     1   2")
    print("    / \\   \\")
    print("   3   4   5")
    print("  /")
    print(" 6")
    
    print_tree_structure(tree, batch_idx=0)
    
    # Find leaf nodes
    leaf_nodes, leaf_edges = find_leaf_nodes(tree)
    print(f"\nLeaf nodes: {leaf_nodes[0]}")
    print(f"Expected leaf nodes: [4, 5, 6] (nodes with degree 1)")
    print(f"Edges connecting to leaves: {leaf_edges[0]}")
    
    # Remove a leaf
    leaf_to_remove = 6
    print(f"\nRemoving leaf node: {leaf_to_remove}")
    modified_tree, removed_edge = remove_leaf_node(tree, batch_idx=0, leaf_node_idx=leaf_to_remove)
    print_tree_structure(modified_tree, batch_idx=0)
    
    print("\nAfter removal:")
    print("       0")
    print("      / \\")
    print("     1   2")
    print("    / \\   \\")
    print("   3   4   5")
    print("  (6 removed)")


if __name__ == "__main__":
    # Run tests
    tree, modified_tree, leaf_nodes, leaf_edges = test_leaf_removal()
    test_with_more_complex_tree()
    
    print("\n" + "=" * 70)
    print("Testing Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Decide how to handle removed edges in CUDA kernels")
    print("  2. Implement edge weight-based pruning")
    print("  3. Test with actual GraphSSM forward pass")
    print("  4. Measure impact on model performance")

