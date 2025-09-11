import os
import sys

# Ensure project root is on sys.path to allow `from core...` when run directly
_HERE = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import networkx as nx

try:
    import matplotlib
    matplotlib.use("Agg")  # headless backend
    import matplotlib.pyplot as plt
except Exception as _matplot_e:  # pragma: no cover
    plt = None
    _MATPLOTLIB_ERR = (
        "Matplotlib not available. To enable graph images, install it:\n"
        "  pip install matplotlib\n"
    )
else:
    _MATPLOTLIB_ERR = None

from core.graph_ssm.tree_utils import prune_tree_by_weight, create_test_case


def _is_connected_undirected(edges: torch.Tensor) -> bool:
    """Check connectivity of an undirected graph given edges [E,2].
    Assumes nodes are labeled with non-negative integers.
    Empty edge set is considered connected only if there is <=1 node.
    """
    if edges.numel() == 0:
        return True

    nodes = edges.view(-1).to(torch.int64)
    unique_nodes = torch.unique(nodes)
    # Build adjacency list
    max_node = nodes.max().item()
    adjacency = [[] for _ in range(max_node + 1)]
    for u, v in edges.tolist():
        adjacency[u].append(v)
        adjacency[v].append(u)

    # BFS/DFS from the first encountered node present in edges
    start = int(unique_nodes[0].item())
    visited = set([start])
    stack = [start]
    while stack:
        curr = stack.pop()
        for nxt in adjacency[curr]:
            if nxt not in visited:
                visited.add(nxt)
                stack.append(nxt)

    # Graph is connected if all nodes that appear in edges are visited
    return visited.issuperset(set(int(n.item()) for n in unique_nodes))


def _save_graph_image(edges: torch.Tensor, title: str, out_path: str):
    """Save a visualization of an undirected graph using NetworkX and Matplotlib."""
    if plt is None:
        print(_MATPLOTLIB_ERR or "")
        return

    edges_np = edges.cpu().numpy().tolist()
    G = nx.Graph()
    for u, v in edges_np:
        G.add_edge(int(u), int(v))

    if len(G) == 0:
        # Empty graph image
        fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
        ax.text(0.5, 0.5, "Empty graph", ha="center", va="center")
        ax.set_axis_off()
    else:
        pos = nx.spring_layout(G, seed=42, k=None)
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        nx.draw_networkx(
            G,
            pos=pos,
            with_labels=True,
            node_color="#1f77b4",
            edge_color="#333333",
            font_size=8,
            node_size=400,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_axis_off()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def test_leaf_only_pruning_preserves_connectivity():
    tree, pairs, weights = create_test_case()
    threshold = 0.7

    pruned = prune_tree_by_weight(tree, pairs, weights, threshold)
    # Remove zero-padding rows if any
    pruned_edges = pruned[0]
    mask = (pruned_edges.sum(dim=1) != 0)
    pruned_edges = pruned_edges[mask]

    assert _is_connected_undirected(pruned_edges), "Leaf-only pruning should not disconnect the tree"

    # Save visualizations
    out_dir = os.path.join(_PROJECT_ROOT, "core", "graph_ssm", "tree_utils_tests", "outputs")
    _save_graph_image(tree[0], "Original (leaf-only case)", os.path.join(out_dir, "leaf_only_original.png"))
    _save_graph_image(pruned_edges, "Pruned leaf-only", os.path.join(out_dir, "leaf_only_pruned.png"))


if __name__ == "__main__":
    # Crude runner: call tests directly and report pass/fail
    failures = []

    def _run(fn):
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except AssertionError as e:
            failures.append((fn.__name__, f"FAIL: {e}"))
            print(f"[FAIL] {fn.__name__}: {e}")
        except Exception as e:
            failures.append((fn.__name__, f"ERROR: {e}"))
            print(f"[ERROR] {fn.__name__}: {e}")

    _run(test_leaf_only_pruning_preserves_connectivity)

    if failures:
        raise SystemExit(1)
    print("All crude tests passed.")

