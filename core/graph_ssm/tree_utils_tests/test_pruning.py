import pytest
import torch

from core.graph_ssm.tree_utils import prune_tree_by_weight
from core.graph_ssm.tree_utils_tests.pruning_test_cases import (
    small_test_case,
    training_test_case,
)


def _safe_draw(edges_tensor: torch.Tensor, file_name: str):
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
    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, file_name)
    plt.savefig(out_path, dpi=200)
    plt.close()


@pytest.mark.parametrize("case_fn", [small_test_case, training_test_case])
def test_pruning_runs(case_fn):
    tree, pairs, weights, threshold = case_fn()
    # Ensure tensors are on CPU for consistency in tests
    tree = tree.to(torch.int32)
    pairs = pairs.to(torch.int32)
    weights = weights.to(torch.float32)

    pruned = prune_tree_by_weight(tree, pairs, weights, threshold)

    assert isinstance(pruned, torch.Tensor)
    assert pruned.dim() == 3 and pruned.size(-1) == 2

    # Draw original and pruned for the first batch element
    case_name = getattr(case_fn, "__name__", "case")
    _safe_draw(tree[0], f"{case_name}_original.png")
    _safe_draw(pruned[0], f"{case_name}_pruned.png")

