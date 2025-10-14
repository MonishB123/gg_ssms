from torch.autograd import Function
from torch.autograd.function import once_differentiable
from tree_scan_lan import _C
import torch
import torch.nn as nn
from einops import rearrange, repeat
import math

# Handle both relative and absolute imports
try:
    from .pruning_utils import find_leaf_nodes_vectorized, prune_leaf_nodes_vectorized
except ImportError:
    from pruning_utils import find_leaf_nodes_vectorized, prune_leaf_nodes_vectorized


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
    weight = -torch.cosine_similarity(fm_ref, fm_tar, dim=-1)  # Fixed: use dim=-1 for consistency
    return torch.exp(weight)  # with - is for min tree


def gaussian_distance(fm_ref, fm_tar, sigma=1.5):
    diff = fm_ref - fm_tar
    weight = (diff * diff).sum(dim=-1) / (2 * sigma * sigma)
    return torch.exp(-weight)  # with - is for max tree

def euclidean_distance(fm_ref, fm_tar):
    diff = fm_ref - fm_tar
    weight = torch.sqrt((diff * diff).sum(dim=-1) + 1e-8)
    return torch.exp(-weight)  # with - is for max tree

def manhattan_distance(fm_ref, fm_tar):
    diff = fm_ref - fm_tar
    weight = torch.abs(diff).sum(dim=-1)
    return torch.exp(-weight)  # with - is for max tree


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

    ### hand-build tree (vectorized)
    # Create chain tree structure: 0->1->2->...->seq_len-1
    tree_indices = torch.arange(seq_len - 1, dtype=torch.int64, device=device)
    tree_ = torch.stack([tree_indices, tree_indices + 1], dim=1)  # [seq_len-1, 2]
    tree = tree_.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch, seq_len-1, 2]
    # Convert to int32 for CUDA kernel compatibility
    tree = tree.int()
    sorted_index1, sorted_parent1, sorted_child1 = bfs(tree, 4)

    ### build tree by feature
    try:
        context_len = min(context_len)
    except:
        context_len = context_len
    
    # Initialize edge mask (will be set if pruning is used)
    edge_mask2 = None
    
    with torch.no_grad():
        def generate_pairs_vectorized(L, prompt_len):
            """Vectorized pair generation for tree construction"""
            pairs = []
            
            # Sequential pairs: 0->1, 1->2, ..., (L-prompt_len-1)->(L-prompt_len)
            if L - prompt_len > 0:
                seq_indices = torch.arange(L - prompt_len, dtype=torch.int64)
                seq_pairs = torch.stack([seq_indices, seq_indices + 1], dim=1)
                pairs.append(seq_pairs)
            
            # Skip connections: (L-prompt_len)->(L-prompt_len+1), (L-prompt_len)->(L-prompt_len+2), etc.
            if L - prompt_len < L - 3:
                start_idx = L - prompt_len
                end_idx = L - 3
                
                # Generate skip connections
                skip_pairs = []
                for i in range(start_idx, end_idx):
                    for skip in range(1, min(4, L - i)):  # Skip 1, 2, or 3 positions
                        skip_pairs.append([i, i + skip])
                
                if skip_pairs:
                    skip_pairs_tensor = torch.tensor(skip_pairs, dtype=torch.int64)
                    pairs.append(skip_pairs_tensor)
            
            # Final connections
            if L >= 3:
                final_pairs = torch.tensor([[L-3, L-2], [L-3, L-1], [L-2, L-1]], dtype=torch.int64)
                pairs.append(final_pairs)
            
            if pairs:
                return torch.cat(pairs, dim=0)
            else:
                return torch.empty((0, 2), dtype=torch.int64)

        if context_len > 2:
            pairs = generate_pairs_vectorized(seq_len, context_len).to(device)
            # Convert to int32 for CUDA kernel compatibility
            pairs = pairs.int()
            data1 = torch.index_select(feature_in, 2, pairs[:, 0])
            data2 = torch.index_select(feature_in, 2, pairs[:, 1])
            
            # MODIFIED: Use the distance function stored in self
            tree_weight = self.distance_fn(data1, data2)

            
            tree = mst(pairs.repeat(batch_size, 1, 1), tree_weight, seq_len)
            
            # Apply dynamic pruning based on actual MST tree structure (vectorized)
            if self.prune_ratio > 0.0:
                # Calculate number of leaves to prune based on actual tree structure
                # Estimate: MST trees typically have ~2 leaf nodes for chain structures
                estimated_leaves = 2  # Conservative estimate for MST trees
                num_leaves_to_prune = max(1, int(estimated_leaves * self.prune_ratio))
                
                # Fix tensor shape mismatch: Extract edge weights for actual tree edges
                num_tree_edges = tree.shape[1]  # Actual number of edges in MST
                if tree_weight.shape[1] >= num_tree_edges:
                    edge_weights_for_pruning = tree_weight[:, :num_tree_edges]
                else:
                    # Handle case where MST has fewer edges than expected
                    edge_weights_for_pruning = torch.ones(batch_size, num_tree_edges, device=tree_weight.device)
                
                # Use vectorized pruning function
                edge_mask2, num_removed_per_batch = prune_leaf_nodes_vectorized(
                    tree, edge_weights_for_pruning, num_leaves_to_prune, verbose=self.verbose
                )
                
                if self.verbose:
                    print(f"Dynamic pruning: {self.prune_ratio:.2%} ratio = {num_leaves_to_prune} leaves to prune")
                    print(f"Edges kept per batch: {edge_mask2.sum(dim=1)}")
                    print(f"Leaves pruned per batch: {num_removed_per_batch}")
            else:
                # No pruning
                if self.verbose:
                    print(f"Pruning disabled (prune_ratio = {self.prune_ratio})")
                edge_mask2 = None
            
            # BFS operates on the full tree (pruning happens via edge weights)
            sorted_index2, sorted_parent2, sorted_child2 = bfs(tree, context_len)
            if self.verbose:
                print(f"sorted_index2 shape: {sorted_index2.shape}")
                print(f"weight shape: {weight.shape}")
        else:
            sorted_index2, sorted_parent2, sorted_child2 = (
                sorted_index1,
                sorted_parent1,
                sorted_child1,
            )

    # Apply pruning to BOTH computation paths for maximum effectiveness
    # Create pruning mask that will be applied to both feature_out1 and feature_out2
    
    # First, compute feature_out1 (unpruned baseline)
    feature_out1 = refine(
        feature_in, weight, sorted_index1, sorted_parent1, sorted_child1
    )
    
    # Prepare edge_weight for feature_out2
    edge_weight = batch_index_opr(weight, sorted_index2)
    
    # Create pruning mask that affects BOTH computation paths
    pruning_mask = None
    if edge_mask2 is not None:
        # Fully vectorized approach - no Python loops, no GPU-CPU sync
        # edge_weight shape: [batch, features, seq_len]
        # edge_mask2 shape: [batch, num_edges] where num_edges = seq_len - 1
        
        batch_size, seq_len = sorted_index2.shape
        num_tree_edges = tree.shape[1]  # Use actual tree edges, not edge_mask2.shape[1]
        
        # Create position mask: start with all positions enabled
        position_mask = torch.ones(batch_size, seq_len, dtype=torch.float32, device=weight.device)
        
        # Fix dimension consistency: Ensure edge_mask2 matches tree dimensions
        if edge_mask2.shape[1] != num_tree_edges:
            if edge_mask2.shape[1] > num_tree_edges:
                edge_mask2 = edge_mask2[:, :num_tree_edges]
            else:
                # Pad with True (keep edges)
                padding = torch.ones(batch_size, num_tree_edges - edge_mask2.shape[1], 
                                   dtype=torch.bool, device=edge_mask2.device)
                edge_mask2 = torch.cat([edge_mask2, padding], dim=1)
        
        # Vectorized approach: Directly map pruned edges to affected positions
        # For MST trees, pruned edges affect their connected nodes
        # We can do this without loops by using advanced indexing
        
        # Get tree structure: [batch, num_tree_edges, 2] -> [batch, num_tree_edges] for src and dst
        tree_src = tree[:, :, 0]  # [batch, num_tree_edges]
        tree_dst = tree[:, :, 1]  # [batch, num_tree_edges]
        
        # Find pruned edges
        pruned_edges = ~edge_mask2  # [batch, num_tree_edges], True where pruned
        
        # Get positions affected by pruned edges (vectorized)
        pruned_src_positions = tree_src[pruned_edges]  # Flattened positions from pruned edges
        pruned_dst_positions = tree_dst[pruned_edges]  # Flattened positions from pruned edges
        
        # Create batch indices for the flattened positions
        batch_indices = torch.arange(batch_size, device=weight.device).unsqueeze(1).expand(-1, num_tree_edges)
        pruned_batch_indices = batch_indices[pruned_edges]
        
        # Zero out affected positions (vectorized)
        # Use advanced indexing to set positions to 0
        valid_src_mask = (pruned_src_positions < seq_len) & (pruned_src_positions >= 0)
        valid_dst_mask = (pruned_dst_positions < seq_len) & (pruned_dst_positions >= 0)
        
        # Apply masks to valid positions only
        if valid_src_mask.any():
            position_mask[pruned_batch_indices[valid_src_mask], pruned_src_positions[valid_src_mask]] = 0.0
        if valid_dst_mask.any():
            position_mask[pruned_batch_indices[valid_dst_mask], pruned_dst_positions[valid_dst_mask]] = 0.0
        
        # Store pruning mask for application to both paths
        pruning_mask = position_mask.unsqueeze(1)  # [batch, 1, seq_len]
        
        # Apply mask to edge_weight for feature_out2
        edge_weight = edge_weight * pruning_mask
        
        if self.verbose:
            num_zeroed = (position_mask == 0).sum(dim=1)
            print(f"Applied pruning mask - zeroed out {num_zeroed} positions per batch")
    
    # Compute feature_out2 (with pruning applied)
    feature_out2 = refine(
        feature_in, edge_weight, sorted_index2, sorted_parent2, sorted_child2
    )
    
    # Apply pruning to feature_out1 as well for maximum effectiveness
    if pruning_mask is not None:
        feature_out1 = feature_out1 * pruning_mask
        if self.verbose:
            print("Applied pruning mask to BOTH computation paths for maximum effectiveness")
    
    # Combine both paths (now both are pruned)
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
        distance_metric='cosine',  # ADDED: distance metric parameter
        prune_ratio=0.15,  # ratio of leaf nodes to prune (0.0 = no pruning, 0.5 = prune 15% of leaves)
        verbose=False,  # whether to print debug information
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.prune_ratio = prune_ratio  # Store pruning ratio
        self.verbose = verbose  # Store verbose flag
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

        # ADDED: Distance metric selection
        self.distance_metric = distance_metric
        self.distance_functions = {
            'cosine': cosine_distance,
            'euclidean': euclidean_distance,
            'gaussian': gaussian_distance,
            'manhattan': manhattan_distance,
            'norm2': norm2_distance,
        }
        
        # Store the selected distance function
        if distance_metric not in self.distance_functions:
            raise ValueError(f"Unknown distance metric: {distance_metric}. "
                           f"Choose from: {list(self.distance_functions.keys())}")
        self.distance_fn = self.distance_functions[distance_metric]
        
        # Dynamic pruning will be computed during forward pass based on actual tree structure
        # No pre-computation needed since pruning depends on actual MST results

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

    # Instantiate the GraphSSM layer with different distance metrics
    print("Testing different distance metrics:")
    for metric in ['cosine', 'euclidean', 'gaussian', 'manhattan', 'norm2']:
        model = GraphSSM(d_model=d_model, distance_metric=metric)
    output = model(x, context_len)
    print(f"  {metric:12s}: Input {x.shape} -> Output {output.shape}")
