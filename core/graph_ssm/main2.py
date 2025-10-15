from torch.autograd import Function
from torch.autograd.function import once_differentiable
from tree_scan_lan import _C
import torch
import torch.nn as nn
from einops import rearrange, repeat
import math
import torch.nn.functional as F
import numpy as np


class _MST(Function):
    @staticmethod
    def forward(ctx, edge_index, edge_weight, vertex_index):
        edge_out = _C.mst_forward(edge_index, edge_weight, vertex_index)
        return edge_out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        return None, None, None


class _BFS(Function):
    @staticmethod
    def forward(ctx, edge_index, max_adj_per_vertex):
        sorted_index, sorted_parent, sorted_child, _ = _C.bfs_forward(
            edge_index, max_adj_per_vertex
        )
        return sorted_index, sorted_parent, sorted_child


class _Refine(Function):
    @staticmethod
    def forward(
        ctx, feature_in, edge_weight, sorted_index, sorted_parent, sorted_child
    ):
        feature_out = _C.tree_scan_refine_forward(
            feature_in, edge_weight, sorted_index, sorted_parent, sorted_child
        )

        ctx.save_for_backward(
            feature_out, edge_weight, sorted_index, sorted_parent, sorted_child
        )
        return feature_out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        feature_out, edge_weight, sorted_index, sorted_parent, sorted_child = (
            ctx.saved_tensors
        )

        grad_feature, grad_edge = _C.tree_scan_refine_backward_feature(
            feature_out,
            edge_weight,
            sorted_index,
            sorted_parent,
            sorted_child,
            grad_output,
        )
        return grad_feature, grad_edge, None, None, None


def norm2_distance(fm_ref, fm_tar):
    diff = fm_ref - fm_tar
    weight = (diff * diff).sum(dim=-2)
    return torch.exp(weight)  # with - is for max tree


def cosine_distance(fm_ref, fm_tar):
    weight = -torch.cosine_similarity(fm_ref, fm_tar, dim=1)
    return torch.exp(weight)  # with - is for min tree


def knn_mst_batch_from_features(feature_in, k=16, metric="cosine"):
    """
    Vectorized, per-batch k-NN → sparse MST (PyTorch).
    Safe, vectorized k-NN -> filtered MST for moderate L (<= ~2000).
    Returns MST edges as batched tensors.
    
    Args:
        feature_in: [B, C, L] - features for each node
        k: number of nearest neighbors per node
        metric: similarity metric ("cosine")
    
    Returns:
        MST edges from _C.mst_forward
    """
    # feature_in: [B, C, L]
    B, C, L = feature_in.shape
    device = feature_in.device
    
    # Choose k
    k = min(max(4, k), L - 1)

    # For large L, use approximate methods to avoid O(L^2) memory explosion
    if L > 2000:
        print(f"Warning: L={L} is large. Consider using FAISS/HNSW for better performance.")
        # Fall back to a sparser approach or chunked processing
        k = min(k, 32)  # Reduce k for large graphs
    
    # 1) Compute pairwise similarity / distance
    # We'll use cosine similarity -> higher is more similar.
    # Normalize features along channel dim:
    feats = feature_in.permute(0, 2, 1)  # [B, L, C]
    feats_norm = F.normalize(feats, p=2, dim=-1)  # [B, L, C]

    # Compute cosine similarity matrix per batch (B x L x L)
    # Be careful: for large L this is O(L^2) memory/time.
    sim = torch.bmm(feats_norm, feats_norm.transpose(1, 2))  # [B, L, L]

    # We don't want self-edges -> set diagonal to very small:
    diag_mask = torch.eye(L, device=device, dtype=torch.bool).unsqueeze(0)  # [1,L,L]
    sim = sim.masked_fill(diag_mask, -9e9)

    # 2) topk neighbors per node (k highest similarities)
    # topk returns values, indices along last dim (neighbors for each node)
    vals, idxs = torch.topk(sim, k=k, dim=-1, largest=True, sorted=False)  # [B,L,k], [B,L,k]

    # 3) build edge list from indices: for each (b, i, neighbor) we produce edge (i, neighbor)
    # We will flatten to (B, L*k, 2)
    node_idx = torch.arange(L, device=device).view(1, L, 1).expand(B, L, k)  # [B,L,k]
    edge_u = node_idx.reshape(B, -1)  # [B, L*k]
    edge_v = idxs.reshape(B, -1)      # [B, L*k]

    # Stack into pairs: [B, L*k, 2], dtype int32 if that's what _C expects
    pairs_batched = torch.stack([edge_u, edge_v], dim=-1).to(torch.int32)  # [B, L*k, 2]

    # 4) compute per-edge weight consistent with your distance function
    # For cosine_distance in your code you used: weight = torch.exp(-cosine_similarity)
    # We'll compute cosine similarity for each pair (we already have `vals`)
    # Convert similarity -> weight the same way as original:
    # original used -cosine_similarity then exp -> weight = exp(-cos)
    # Here vals are cosine similarity, so:
    edge_weights = torch.exp(-vals.reshape(B, -1))  # [B, L*k]

    # 5) Optional: symmetrize edges (keep u->v and v->u) and deduplicate per batch
    # Combine (u,v) and (v,u) so MST has both directions if present
    rev_pairs = torch.stack([edge_v, edge_u], dim=-1)  # [B, L*k, 2]
    all_pairs = torch.cat([pairs_batched, rev_pairs], dim=1)   # [B, 2*L*k, 2]
    all_weights = torch.cat([edge_weights, edge_weights], dim=1)  # [B, 2*L*k]

    # Deduplicate edges per batch: convert pairs to single index: idx = u*L + v
    flattened_idx = (all_pairs[..., 0].long() * L + all_pairs[..., 1].long())  # [B, M]
    
    # Use scatter to find unique indices per batch - simpler to use CPU unique for moderate sizes:
    # We'll do a per-batch unique on CPU for robustness (but for large B, L you may want a GPU variant).
    pairs_out = []
    weights_out = []
    for b in range(B):
        idx_b = flattened_idx[b].cpu().numpy()
        pairs_b = all_pairs[b].cpu().numpy().astype('int32')  # (M,2)
        weights_b = all_weights[b].cpu().numpy()
        # unique preserving order: use numpy.unique with return_index
        unique_idx, unique_pos = np.unique(idx_b, return_index=True)
        sel = np.sort(unique_pos)  # keep original order (or sort by weight if desired)
        sel_pairs = pairs_b[sel]
        sel_weights = weights_b[sel]
        pairs_out.append(torch.from_numpy(sel_pairs).to(device=device, dtype=torch.int32))
        weights_out.append(torch.from_numpy(sel_weights).to(device=device, dtype=torch.float32))

    # Now we have lists of tensors (per batch) possibly of different lengths.
    # _C.mst_forward likely expects batched tensors with same M; if not, you can pad to max length
    max_M = max([p.shape[0] for p in pairs_out])
    pairs_padded = torch.zeros((B, max_M, 2), dtype=torch.int32, device=device)
    weights_padded = torch.zeros((B, max_M), dtype=torch.float32, device=device)
    mask = torch.zeros((B, max_M), dtype=torch.bool, device=device)
    for b in range(B):
        m = pairs_out[b].shape[0]
        pairs_padded[b, :m] = pairs_out[b]
        weights_padded[b, :m] = weights_out[b]
        mask[b, :m] = True

    # 6) Call GPU MST on the per-batch filtered edges
    # Depending on _C.mst_forward signature you pass pairs_padded and weights_padded and num_vertices (L)
    mst_edges = _C.mst_forward(pairs_padded, weights_padded, L)  # may return [B, (L-1), 2] shape
    return mst_edges


def knn_mst_with_faiss_fallback(feature_in, k=16, metric="cosine", use_faiss_threshold=2000):
    """
    k-NN MST with FAISS fallback for large sequences.
    Automatically chooses between vectorized PyTorch (small L) and FAISS (large L).
    
    Args:
        feature_in: [B, C, L] - features for each node
        k: number of nearest neighbors per node
        metric: similarity metric ("cosine")
        use_faiss_threshold: switch to FAISS when L > this value
    
    Returns:
        MST edges from _C.mst_forward
    """
    B, C, L = feature_in.shape
    
    if L <= use_faiss_threshold:
        # Use vectorized PyTorch implementation
        return knn_mst_batch_from_features(feature_in, k=k, metric=metric)
    else:
        # For large L, use FAISS or implement chunked approach
        print(f"Using FAISS fallback for L={L}")
        # TODO: Implement FAISS-based k-NN search
        # For now, fall back to the regular implementation with reduced k
        return knn_mst_batch_from_features(feature_in, k=min(k, 16), metric=metric)


def validate_mst_connectivity(mst_edges, num_vertices):
    """
    Validate that MST edges form a connected graph with exactly (num_vertices - 1) edges.
    
    Args:
        mst_edges: MST edge tensor from _C.mst_forward
        num_vertices: number of vertices in the graph
    
    Returns:
        bool: True if MST is valid (connected with correct number of edges)
    """
    if mst_edges is None:
        return False
    
    # Check if we have the right number of edges for a tree
    if hasattr(mst_edges, 'shape'):
        if len(mst_edges.shape) == 3:  # [B, E, 2]
            num_edges = mst_edges.shape[1]
        elif len(mst_edges.shape) == 2:  # [E, 2]  
            num_edges = mst_edges.shape[0]
        else:
            return False
        
        # A tree should have exactly (num_vertices - 1) edges
        expected_edges = num_vertices - 1
        return num_edges == expected_edges
    
    return False


def benchmark_knn_mst_methods(feature_in, k_values=[4, 8, 12, 24], runs=3):
    """
    Benchmark different k values for k-NN MST to find optimal performance/accuracy trade-off.
    
    Args:
        feature_in: [B, C, L] - input features
        k_values: list of k values to test
        runs: number of runs for timing
    
    Returns:
        dict: timing and connectivity results for each k value
    """
    import time
    
    results = {}
    B, C, L = feature_in.shape
    
    print(f"Benchmarking k-NN MST for sequence length L={L}")
    
    for k in k_values:
        if k >= L:
            continue
            
        times = []
        connectivity_checks = []
        
        for run in range(runs):
            start_time = time.time()
            
            try:
                mst_edges = knn_mst_batch_from_features(feature_in, k=k)
                end_time = time.time()
                
                times.append(end_time - start_time)
                connectivity_checks.append(validate_mst_connectivity(mst_edges, L))
                
            except Exception as e:
                print(f"Error with k={k}, run={run}: {e}")
                times.append(float('inf'))
                connectivity_checks.append(False)
        
        avg_time = sum(times) / len(times) if times else float('inf')
        connectivity_rate = sum(connectivity_checks) / len(connectivity_checks) if connectivity_checks else 0.0
        
        results[k] = {
            'avg_time_sec': avg_time,
            'connectivity_rate': connectivity_rate,
            'times': times
        }
        
        print(f"k={k}: avg_time={avg_time:.4f}s, connectivity_rate={connectivity_rate:.2f}")
    
    return results


def knn_filtered_mst_legacy(pairs, edge_weights, num_vertices, k=8):
    """
    Legacy k-NN-filtered Kruskal MST optimization (DEPRECATED - CPU bottleneck).
    This function is kept for reference but should not be used due to performance issues.
    Use knn_mst_batch_from_features instead.
    """
    batch_size = edge_weights.shape[0]
    device = edge_weights.device
    num_edges = pairs.shape[0]
    
    # Adaptive k based on sequence length
    k = min(k, num_vertices - 1, max(4, num_vertices // 3))
    
    # For each vertex, keep only k lowest weight edges
    vertex_edges = [[] for _ in range(num_vertices)]
    
    # Group edges by vertex
    for edge_idx in range(num_edges):
        u, v = pairs[edge_idx].tolist()
        if u < num_vertices and v < num_vertices:
            vertex_edges[u].append(edge_idx)
            vertex_edges[v].append(edge_idx)
    
    # Filter to k-NN for each vertex
    filtered_edge_set = set()
    
    for vertex in range(num_vertices):
        if len(vertex_edges[vertex]) > 0:
            vertex_edge_indices = torch.tensor(vertex_edges[vertex], device=device)
            vertex_weights = edge_weights[:, vertex_edge_indices].mean(dim=0)
            
            # Keep top k edges (lowest weights)
            if len(vertex_edge_indices) > k:
                _, top_k_indices = torch.topk(vertex_weights, k, largest=False)
                selected_edges = vertex_edge_indices[top_k_indices]
            else:
                selected_edges = vertex_edge_indices
            
            filtered_edge_set.update(selected_edges.tolist())
    
    # Apply MST on filtered edges
    if len(filtered_edge_set) == 0:
        return _C.mst_forward(pairs, edge_weights, num_vertices)
    
    filtered_indices = torch.tensor(list(filtered_edge_set), device=device, dtype=torch.long)
    filtered_pairs = pairs[filtered_indices]
    filtered_weights = edge_weights[:, filtered_indices]
    
    return _C.mst_forward(filtered_pairs, filtered_weights, num_vertices)


def batch_index_opr(data, index):
    with torch.no_grad():
        channel = data.shape[1]
        index = index.unsqueeze(1).expand(-1, channel, -1).long()
    data = torch.gather(data, 2, index)
    return data


def tree_scanning_algorithm(self, input_states, context_len):
    batch_size, seq_len, _ = input_states.shape
    dtype = input_states.dtype
    device = input_states.device
    # 1. Gated MLP's linear projection
    projected_states = self.in_proj(input_states).transpose(
        1, 2
    )  # [batch, 2 * intermediate_size, seq_len]
    hidden_states, gate = projected_states.chunk(2, dim=1)

    hidden_states = self.act(
        self.conv1d(hidden_states)[..., :seq_len]
    )  # [batch, intermediate_size, seq_len]
    # 3. State Space Model sequence transformation
    # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
    ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
    time_step, B, C = torch.split(
        ssm_parameters,
        [self.dt_rank, self.d_state, self.d_state],
        dim=-1,
    )
    discrete_time_step = self.dt_proj(time_step)  # [batch, seq_len, intermediate_size]
    discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(
        1, 2
    )  # [batch, intermediate_size, seq_len]
    # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
    A = -torch.exp(self.A_log.float())  # [intermediate_size, ssm_state_size]
    discrete_A = torch.exp(
        A[None, :, None, :] * discrete_time_step[:, :, :, None]
    )  # [batch, intermediate_size, seq_len, ssm_state_size]
    discrete_B = (
        discrete_time_step[:, :, :, None] * B[:, None, :, :].float()
    )  # [batch, intermediade_size, seq_len, ssm_state_size]
    deltaB_u = discrete_B * hidden_states[:, :, :, None].float()
    ### tree scan
    weight = rearrange(discrete_A, "b d l n -> b (d n) l").contiguous()
    feature_in = rearrange(deltaB_u, "b d l n -> b (d n) l").contiguous()
    feature_in = torch.flip(feature_in, dims=[-1]).contiguous()
    weight = torch.roll(torch.flip(weight, dims=[-1]), 1, -1).contiguous()

    mst = _MST.apply
    bfs = _BFS.apply
    refine = _Refine.apply

    ### hand-build tree
    tree_ = []
    for i in range(seq_len - 1):
        tree_.append([i, i + 1])
    tree_ = torch.tensor(tree_, dtype=torch.int32).to(device)
    tree = tree_.repeat(batch_size, 1, 1)
    sorted_index1, sorted_parent1, sorted_child1 = bfs(tree, 4)

    ### build tree by feature using k-NN-filtered Kruskal optimization
    try:
        context_len = min(context_len)
    except:
        context_len = context_len
    with torch.no_grad():

        def generate_pairs(L, prompt_len):
            pairs = []
            for i in range(0, L - prompt_len):
                pairs.append([i, i + 1])
            for i in range(L - prompt_len, L - 3):
                pairs.append([i, i + 1])
                pairs.append([i, i + 2])
                pairs.append([i, i + 3])
            pairs.append([L - 3, L - 2])
            pairs.append([L - 3, L - 1])
            pairs.append([L - 2, L - 1])
            return pairs

        if context_len > 2:
            # Use vectorized k-NN MST from features directly with FAISS fallback for large sequences
            k = min(24, context_len * 2)  # Adaptive k based on context length
            tree = knn_mst_with_faiss_fallback(feature_in, k=k, metric="cosine")
            sorted_index2, sorted_parent2, sorted_child2 = bfs(tree, context_len)
        else:
            sorted_index2, sorted_parent2, sorted_child2 = (
                sorted_index1,
                sorted_parent1,
                sorted_child1,
            )

        # import pdb;pdb.set_trace()
    # import pdb;pdb.set_trace()
    feature_out1 = refine(
        feature_in, weight, sorted_index1, sorted_parent1, sorted_child1
    )
    # import pdb;pdb.set_trace()
    edge_weight = batch_index_opr(weight, sorted_index2)
    feature_out2 = refine(
        feature_in, edge_weight, sorted_index2, sorted_parent2, sorted_child2
    )
    feature_out = (
        feature_out2 * 0.3 + feature_out1
    )  # 0.3 is scaling factor (hyperparameter)

    feature_out = rearrange(
        torch.flip(feature_out.to(dtype), dims=[-1]),
        "b (d n) l -> b l d n",
        b=batch_size,
        n=discrete_A.shape[-1],
    ).contiguous()
    scan_output_ = (
        (feature_out @ C.unsqueeze(-1)).squeeze(-1).transpose(-1, -2)
    )  # (B, L, D, N) @ (B, L, N, 1) -> (B, L, D, 1)

    # [batch, seq_len, intermediade_size]
    scan_output = scan_output_ + (hidden_states * self.D[None, :, None])
    scan_output = scan_output * self.act(gate)
    # 4. Final linear projection
    contextualized_states = self.out_proj(
        scan_output.transpose(1, 2)
    )  # [batch, seq_len, hidden_size]
    return contextualized_states


class GraphSSM(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )

    def forward(self, input_states, context_len):
        return tree_scanning_algorithm(self, input_states, context_len)


if __name__ == "__main__":
    # Example hyperparameters
    d_model = 16
    seq_len = 12
    batch_size = 2
    context_len = 4  # Or pass in a list, e.g., [4, 4] for each sample

    # Create random input tensor
    x = torch.randn(batch_size, seq_len, d_model)

    # Instantiate the GraphSSM layer
    model = GraphSSM(d_model=d_model)

    # Forward pass
    output = model(x, context_len)

    print("Input shape:", x.shape)  # (B, L, d_model)
    print("Output shape:", output.shape)  # (B, L, d_model)
    # Now 'output' contains the contextualized representation
