import torch
import os
import sys

# Add the parent directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tree_utils import prune_tree_by_weight
from tree_utils_tests.pruning_test_cases import (
    small_test_case,
    training_test_case,
)


def _save_edge_info(tree: torch.Tensor, pairs: torch.Tensor, weights: torch.Tensor, 
                   pruned: torch.Tensor, threshold: float, case_dir: str):
    """Save original and pruned edge information to text files."""
    base_dir = os.path.dirname(__file__)
    out_dir = os.path.join(base_dir, "outputs", case_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    # Process only the first batch for simplicity
    if tree.shape[0] > 0:
        batch_idx = 0
        tree_b = tree[batch_idx]
        weights_b = weights[batch_idx]
        pruned_b = pruned[batch_idx]
        
        # Create a mapping from edge pairs to weights
        pair_to_weight = {}
        for i, (u, v) in enumerate(pairs.tolist()):
            pair_to_weight[(u, v)] = weights_b[i].item()
            pair_to_weight[(v, u)] = weights_b[i].item()  # bidirectional
        
        # Save raw data to file
        analysis_file = os.path.join(out_dir, "edge_analysis.md")
        with open(analysis_file, 'w') as f:
            # Get original MST edges and weights
            original_edges_list = [e for e in tree_b.tolist() if not (e[0] == 0 and e[1] == 0)]
            original_weights_list = [round(pair_to_weight.get((u, v), 0.0), 4) for u, v in original_edges_list]
            
            # Get pruned MST edges and weights
            pruned_edges_list = [e for e in pruned_b.tolist() if not (e[0] == 0 and e[1] == 0)]
            pruned_weights_list = [round(pair_to_weight.get((u, v), 0.0), 4) for u, v in pruned_edges_list]
            
            # Create markdown table: Original edges, Original weights, Pruned edges, Pruned weights
            f.write("| Original Edges | Original Weights | Pruned Edges | Pruned Weights |\n")
            f.write("|----------------|------------------|--------------|----------------|\n")
            
            # Find the maximum length to pad shorter lists
            max_len = max(len(original_edges_list), len(pruned_edges_list))
            
            # Pad shorter list with empty strings
            original_edges_padded = original_edges_list + [""] * (max_len - len(original_edges_list))
            original_weights_padded = original_weights_list + [""] * (max_len - len(original_weights_list))
            pruned_edges_padded = pruned_edges_list + [""] * (max_len - len(pruned_edges_list))
            pruned_weights_padded = pruned_weights_list + [""] * (max_len - len(pruned_weights_list))
            
            # Write each row
            for i in range(max_len):
                f.write(f"| {original_edges_padded[i]} | {original_weights_padded[i]} | {pruned_edges_padded[i]} | {pruned_weights_padded[i]} |\n")
        
        print(f"\nSaved raw edge data to: {analysis_file}")


def _safe_draw(edges_tensor: torch.Tensor, case_dir: str, file_name: str):
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        import networkx as nx
    except Exception:
        return  # Drawing is optional; skip if libs not available

    # Convert to list of (u, v), filtering padding/self-loops
    edges_np = edges_tensor.detach().cpu().numpy()
    edges = [(int(u), int(v)) for u, v in edges_np if not (int(u) == 0 and int(v) == 0) and int(u) != int(v)]
    if len(edges) == 0:
        return

    G = nx.Graph()
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, seed=0)

    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, with_labels=True, node_size=100, font_size=8)
    plt.tight_layout()

    import os
    base_dir = os.path.dirname(__file__)
    out_dir = os.path.join(base_dir, "outputs", case_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, file_name)
    plt.savefig(out_path, dpi=200)
    plt.close()


def test_pruning_runs(case_fn):
    tree, pairs, weights, threshold = case_fn()
    # Ensure tensors are on CPU for consistency in tests
    tree = tree.to(torch.int32)
    pairs = pairs.to(torch.int32)
    weights = weights.to(torch.float32)

    pruned = prune_tree_by_weight(tree, pairs, weights, threshold)

    assert isinstance(pruned, torch.Tensor)
    assert pruned.dim() == 3 and pruned.size(-1) == 2

    # Save edge information to files
    case_name = getattr(case_fn, "__name__", "case")
    _save_edge_info(tree, pairs, weights, pruned, threshold, case_name)

    # Draw original and pruned for all batch elements
    B = tree.shape[0]
    for b in range(B):
        _safe_draw(tree[b], case_name, f"b{b:02d}_original.png")
        _safe_draw(pruned[b], case_name, f"b{b:02d}_pruned.png")


if __name__ == "__main__":
    # Run tests for both cases
    print("Running pruning tests...")
    
    print("\n=== Testing small_test_case ===")
    test_pruning_runs(small_test_case)
    
    print("\n=== Testing training_test_case ===")
    test_pruning_runs(training_test_case)
    
    print("\nAll tests completed!")

