import torch
import admin_torch

from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import unbatch_edge_index
from geomloss import SamplesLoss

from utils.norm_layers import BN, LN
from esa.mha import SAB, PMA
from esa.mlp_utils import SmallMLP, GatedMLPMulti


def get_adj_mask_from_edge_index_node(
    edge_index, batch_size, max_items, batch_mapping, xformers_or_torch_attn, use_bfloat16=True, device="cuda:0"
):
    if xformers_or_torch_attn in ["torch"]:
        empty_mask_fill_value = False
        mask_dtype = torch.bool
        edge_mask_fill_value = True
    else:
        empty_mask_fill_value = -99999
        mask_dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        edge_mask_fill_value = 0

    adj_mask = torch.full(
        size=(batch_size, max_items, max_items),
        fill_value=empty_mask_fill_value,
        device=device,
        dtype=mask_dtype,
        requires_grad=False,
    )

    edge_index_unbatched = unbatch_edge_index(edge_index, batch_mapping)
    edge_batch_non_cumulative = torch.cat(edge_index_unbatched, dim=1)

    edge_batch_mapping = batch_mapping.index_select(0, edge_index[0, :])

    adj_mask[
        edge_batch_mapping, edge_batch_non_cumulative[0, :], edge_batch_non_cumulative[1, :]
    ] = edge_mask_fill_value

    if xformers_or_torch_attn in ["torch"]:
        adj_mask = ~adj_mask

    adj_mask = adj_mask.unsqueeze(1)
    return adj_mask


def create_edge_adjacency_mask(edge_index, num_edges):
    # Get all the nodes in the edge index (source and target separately for undirected graphs)
    source_nodes = edge_index[0]
    target_nodes = edge_index[1]

    # Create expanded versions of the source and target node tensors
    expanded_source_nodes = source_nodes.unsqueeze(1).expand(-1, num_edges)
    expanded_target_nodes = target_nodes.unsqueeze(1).expand(-1, num_edges)

    # Create the adjacency mask where an edge is adjacent if either node matches either node of other edges
    source_adjacency = expanded_source_nodes == expanded_source_nodes.t()
    target_adjacency = expanded_target_nodes == expanded_target_nodes.t()
    cross_adjacency = (expanded_source_nodes == expanded_target_nodes.t()) | (
        expanded_target_nodes == expanded_source_nodes.t()
    )

    adjacency_mask = source_adjacency | target_adjacency | cross_adjacency

    # Mask out self-adjacency by setting the diagonal to False
    adjacency_mask.fill_diagonal_(0)  # We use "0" here to indicate False in PyTorch boolean context

    return adjacency_mask



def expand_hyperedge_adjacency_mask(hyperedge_index, edge_adj_matrix, edge_index, num_edges):
    # initialization
    num_hyperedges = hyperedge_index.size(0) ; total_num_edges = num_edges + num_hyperedges
    expanded_adj = torch.zeros((total_num_edges, total_num_edges), dtype=torch.bool, device=edge_adj_matrix.device)
    expanded_adj[:num_edges, :num_edges] = edge_adj_matrix
    

    # hyperedge - hyperedge connectivity
    source_nodes = edge_index[0].long().flatten() ; target_nodes = edge_index[1].long().flatten()
    sentinel = max(edge_index.max().item() + 2, hyperedge_index.max().item() + 2, source_nodes.max()+2, target_nodes.max()+2)  # A value that is not a valid node index
    hed = hyperedge_index.clone() ; hed[hed == -1] = sentinel  # [H, K]
    num_hyperedges, max_nodes = hed.shape # Create a mask of shape [H, N] where N = max node index + 1
    num_nodes = sentinel + 1 

    node_mask = torch.zeros((num_hyperedges, num_nodes), dtype=torch.bool, device=edge_adj_matrix.device)
    node_mask.scatter_(1, hed, True) ; node_mask = node_mask[:, :-1]

    node_mask_float = node_mask.float()   # [H, N] # Remove sentinel column so padding doesn't count as overlap
    hyperedge_connectivity = ((node_mask_float @ node_mask_float.T) > 0).int()  # [H, H] True if any shared node # Compute hyperedge-hyperedge overlaps via matrix multiplication
    hyperedge_connectivity.fill_diagonal_(0)  # Remove self-connections
    expanded_adj[num_edges:, num_edges:] = hyperedge_connectivity


    # edge - hyperedge connectivity
    source_nodes = edge_index[0].long().flatten() ; target_nodes = edge_index[1].long().flatten()


    edge_hyperedge_connectivity = ((node_mask[:, source_nodes] | node_mask[:, target_nodes]).T).int() # [E, H]
    # edge_hyperedge_connectivity = (node_mask[:, source_nodes.T] | node_mask[:, target_nodes.T]).T  # [E, H] = edge connects to hyperedge if source or target node in hyperedge
    expanded_adj[:num_edges, num_edges:] = edge_hyperedge_connectivity 
    expanded_adj[num_edges:, :num_edges] = edge_hyperedge_connectivity.T  
    # edge_to_hyper = node_mask[:, target_nodes.T].T  ; hyper_to_edge = node_mask[:, source_nodes.T] ; #expanded_adj[:num_edges, num_edges:] = edge_to_hyper ; expanded_adj[num_edges:, :num_edges] = hyper_to_edge
    # need to check the diagonal by block aspect
    return expanded_adj



     

def get_first_unique_index(t):
    # This is taken from Stack Overflow :)
    unique, idx, counts = torch.unique(t, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)

    zero = torch.tensor([0], device=torch.device("cuda:0"))
    cum_sum = torch.cat((zero, cum_sum[:-1]))
    first_indicies = ind_sorted[cum_sum]

    return first_indicies


def generate_consecutive_tensor(input_tensor, final):
    lengths = input_tensor[1:] - input_tensor[:-1]  # Calculate the length of each segment
    lengths = torch.cat((lengths, torch.tensor([final - input_tensor[-1]], device=torch.device("cuda:0"))))  # Append the final length
    ranges = [torch.arange(0, length, device=torch.device("cuda:0")) for length in lengths]  # Create ranges for each segment
    result = torch.cat(ranges) # Concatenate all ranges into a single tensor
    return result

# This is needed if the standard "nonzero" method from PyTorch fails
# This alternative is slower but allows bypassing the problem until 64-bit
# support is available
def nonzero_chunked(ten, num_chunks):
    # This is taken from this pull request
    # https://github.com/facebookresearch/segment-anything/pull/569/files
    b, w_h = ten.shape
    total_elements = b * w_h
    # Maximum allowable elements in one chunk - as torch is using 32 bit integers for this function
    max_elements_per_chunk = 2**31 - 1
    # Calculate the number of chunks needed
    if total_elements % max_elements_per_chunk != 0:
        num_chunks += 1
    # Calculate the actual chunk size
    chunk_size = b // num_chunks
    if b % num_chunks != 0:
        chunk_size += 1

    # List to store the results from each chunk
    all_indices = []
    # Loop through the diff tensor in chunks
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, b)
        chunk = ten[start:end, :]
        # Get non-zero indices for the current chunk
        indices = chunk.nonzero()
        # Adjust the row indices to the original tensor
        indices[:, 0] += start
        all_indices.append(indices)

    # Concatenate all the results
    change_indices = torch.cat(all_indices, dim=0)

    return change_indices


def get_adj_mask_from_edge_index_edge(
    edge_index,
    batch_size,
    max_items,
    batch_mapping,
    xformers_or_torch_attn,
    use_bfloat16=True,
    device="cuda:0",
    is_using_hyperedges=False,
    hyperedge_index=None,
    hedge_batch_index=None,
):
    if xformers_or_torch_attn in ["torch"]:
        empty_mask_fill_value = False
        mask_dtype = torch.bool 
        edge_mask_fill_value = True
    else:
        empty_mask_fill_value = -99999
        mask_dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        edge_mask_fill_value = 0
    
    adj_mask = torch.full(
        size=(batch_size, max_items, max_items),
        fill_value=empty_mask_fill_value,
        device=device,
        dtype=mask_dtype,
        requires_grad=False,
    )

    edge_batch_mapping = batch_mapping.index_select(0, edge_index[0, :])
    if is_using_hyperedges and hedge_batch_index is not None:
        edge_batch_mapping = hedge_batch_index

    edge_adj_matrix = create_edge_adjacency_mask(edge_index, edge_index.shape[1])
    if is_using_hyperedges and hyperedge_index is not None:
        edge_adj_matrix = expand_hyperedge_adjacency_mask(hyperedge_index, edge_adj_matrix, edge_index, edge_index.shape[1])


    edge_batch_index_to_original_index = generate_consecutive_tensor(
        get_first_unique_index(edge_batch_mapping), edge_batch_mapping.shape[0]
    )

    try:
        eam_nonzero = edge_adj_matrix.nonzero()
    except:
        # Adjust chunk size as desired
        eam_nonzero = nonzero_chunked(edge_adj_matrix, 3)

    adj_mask[
        edge_batch_mapping.index_select(0, eam_nonzero[:, 0]),
        edge_batch_index_to_original_index.index_select(0, eam_nonzero[:, 0]),
        edge_batch_index_to_original_index.index_select(0, eam_nonzero[:, 1]),
    ] = edge_mask_fill_value

    if xformers_or_torch_attn in ["torch"]:
        adj_mask = ~adj_mask
    
    adj_mask = adj_mask.unsqueeze(1)
    return adj_mask




def batched_effective_rank(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # Compute singular values: shape (B, min(N, D))
    S = torch.linalg.svdvals(X)
    # Normalize singular values to get probability vectors
    S_sum = S.sum(dim=1, keepdim=True)  # shape (B, 1)
    mask = S_sum > eps  # to handle all-zero matrices
    p = S / (S_sum + eps)  # shape (B, r)
    # Compute entropy per batch
    entropy = -torch.sum(p * torch.log(p + eps), dim=1)  # shape (B,)
    # Effective rank = exp(entropy)
    eff_rank = torch.exp(entropy)
    # Zero out effective rank where total singular value was too small
    eff_rank = eff_rank * mask.squeeze(1)

    return eff_rank








# def compute_sinkhorn_distance_matrix(X, p=2, blur=0.005):
#     """
#     Compute symmetric pairwise Sinkhorn Wasserstein distance matrix.
#     Args:   X (torch.Tensor): A tensor of shape (B, E, d) representing a batch of B point clouds, each with E points in d dimensions.
#             p (int): The exponent in the cost function (default: 2).
#             blur (float): The blur parameter for Sinkhorn (default: 0.005).
#     Returns: torch.Tensor: A (B, B) symmetric matrix of pairwise Sinkhorn distances.
#     """
#     B, E, _ = X.shape ; device = X.device
#     a = torch.ones(E, device=device) / E # Uniform weights
#     loss_fn = SamplesLoss("sinkhorn", p=p, blur=blur)# Sinkhorn loss
#     D = torch.zeros(B, B, device=device)# Distance matrix

#     for i in range(B):
#         for j in range(i + 1, B):
#             dist = loss_fn(a, X[i], a, X[j])
#             D[i, j] = dist ; D[j, i] = dist  # symmetric
#     return D


def compute_sinkhorn_distance_matrix(X, p=2, blur=0.05, scaling=0.95):
    """
    Fast, symmetric pairwise Sinkhorn Wasserstein distance matrix.
    Args:  X: (B, E, d) tensor of point clouds
    Returns: D: (B, B) symmetric matrix of Sinkhorn distances
    """
    B, E, d = X.shape ; device = X.device

    a = torch.ones(E, device=device) / E
    loss_fn = SamplesLoss("sinkhorn", p=p, blur=blur, scaling=0.9, backend="tensorized")
    idx_i, idx_j = torch.triu_indices(B, B, offset=1)# Create Cartesian index pairs (i, j) for i < j
    # Flattened input for vectorized loss computation # shape: (num_pairs, E, d)
    X_i = X[idx_i] ; X_j = X[idx_j]  
    dist = loss_fn(a.expand(X_i.shape[0], -1), X_i, a.expand(X_j.shape[0], -1), X_j)

    # Fill symmetric distance matrix
    D = torch.zeros(B, B, device=device)
    D[idx_i, idx_j] = dist ; D[idx_j, idx_i] = dist
    return D


# def center_gram(K):
#     n = K.size(0)
#     H = torch.eye(n, device=K.device) - (1.0 / n) * torch.ones((n, n), device=K.device)
#     return H @ K @ H

# def gram_linear(X):
#     return X @ X.T

# def linear_CKA(X, Y):
#     K = center_gram(gram_linear(X)) ; L = center_gram(gram_linear(Y))
#     hsic = (K * L).sum()
#     norm_K = (K * K).sum().sqrt() ; norm_L = (L * L).sum().sqrt()
#     return hsic / (norm_K * norm_L)


def linear_CKA(X, Y, eps=1e-8):
    """
    Compute linear CKA between two feature matrices X and Y.
    Args: X, Y: (n x d1), (n x d2) input feature matrices (rows = samples)
        eps: numerical stability epsilon
    Returns: cka: scalar similarity in [0, 1]
    """
    K = X @ X.T ;  L = Y @ Y.T  # Compute Gram (dot product) matrices
    K_mean_row = K.mean(dim=0, keepdim=True) ; K_mean_col = K.mean(dim=1, keepdim=True) ; K_mean = K.mean()
    K_centered = K - K_mean_row - K_mean_col + K_mean

    L_mean_row = L.mean(dim=0, keepdim=True) ; L_mean_col = L.mean(dim=1, keepdim=True) ; L_mean = L.mean()
    L_centered = L - L_mean_row - L_mean_col + L_mean

    hsic = (K_centered * L_centered).sum()# Compute HSIC numerator
    norm_K = (K_centered * K_centered).sum().sqrt() ; norm_L = (L_centered * L_centered).sum().sqrt()
    return hsic / (norm_K * norm_L + eps)









class SABComplete(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        num_heads,
        dropout,
        idx,
        norm_type,
        use_mlp=False,
        mlp_hidden_size=64,
        mlp_type="standard",
        node_or_edge="edge",
        xformers_or_torch_attn="xformers",
        residual_dropout=0,
        set_max_items=0,
        use_bfloat16=True,
        num_mlp_layers=3,
        pre_or_post="pre",
        num_layers_for_residual=0,
        use_mlp_ln=False,
        mlp_dropout=0,
    ):
        super(SABComplete, self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.use_mlp = use_mlp
        self.idx = idx
        self.mlp_hidden_size = mlp_hidden_size
        self.node_or_edge = node_or_edge
        self.xformers_or_torch_attn = xformers_or_torch_attn
        self.residual_dropout = residual_dropout
        self.set_max_items = set_max_items
        self.use_bfloat16 = use_bfloat16
        self.num_mlp_layers = num_mlp_layers
        self.pre_or_post = pre_or_post

        if self.pre_or_post == "post":
            self.residual_attn = admin_torch.as_module(num_layers_for_residual)
            self.residual_mlp = admin_torch.as_module(num_layers_for_residual)

        if dim_in != dim_out:
            self.proj_1 = nn.Linear(dim_in, dim_out)
    

        self.sab = SAB(dim_in, dim_out, num_heads, dropout, xformers_or_torch_attn)

        if self.idx != 2:
            bn_dim = self.set_max_items
        else:
            bn_dim = 32
 
        if norm_type == "LN":
            if self.pre_or_post == "post":
                if self.idx != 2:
                    self.norm = LN(dim_out, num_elements=self.set_max_items)
                else:
                    self.norm = LN(dim_out)
            else:
                if self.idx != 2:
                    self.norm = LN(dim_in, num_elements=self.set_max_items)
                else:
                    self.norm = LN(dim_in)
                    
        elif norm_type == "BN":
            self.norm = BN(bn_dim)

        self.mlp_type = mlp_type

        if self.use_mlp:
            if self.mlp_type == "standard":
                self.mlp = SmallMLP(
                    in_dim=dim_out,
                    out_dim=dim_out,
                    inter_dim=mlp_hidden_size,
                    dropout_p=mlp_dropout,
                    num_layers=num_mlp_layers,
                    use_ln=use_mlp_ln,
                )
            
            elif self.mlp_type == "gated_mlp":
                self.mlp = GatedMLPMulti(
                    in_dim=dim_out,
                    out_dim=dim_out,
                    inter_dim=mlp_hidden_size,
                    dropout_p=mlp_dropout,
                    num_layers=num_mlp_layers,
                    use_ln=use_mlp_ln,
                )

        if norm_type == "LN":
            if self.idx != 2:
                self.norm_mlp = LN(dim_out, num_elements=self.set_max_items)
            else:
                self.norm_mlp = LN(dim_out)
                
        elif norm_type == "BN":
            self.norm_mlp = BN(bn_dim)


    def forward(self, inp):
        X, edge_index, batch_mapping, max_items, adj_mask, hidden_adj_masked, hidden_adj_self, zero_mask = inp
        if self.pre_or_post == "pre":
            X = self.norm(X) # also work when hyperedges in x
        
        # print("before sab", X[zero_mask])
        if self.idx == 1:
            if hidden_adj_masked is not None:
                out_attn = self.sab(X, hidden_adj_masked)
            else :
                out_attn = self.sab(X, adj_mask)
        else:
            if hidden_adj_self is not None :
                out_attn = self.sab(X, hidden_adj_self)
            else:
                out_attn = self.sab(X)
        
        out_attn[zero_mask] = 0
        if out_attn.shape[-1] != X.shape[-1]:
            X = self.proj_1(X)

        if self.pre_or_post == "pre":
            out = X + out_attn
            out[zero_mask] = 0
        if self.pre_or_post == "post":
            out = self.residual_attn(X, out_attn)

            b, max_num , f = X.shape
            # Split the tensor into two parts
            if hidden_adj_masked is not None:
                max_num_divided = max_num//2
                out1 = out[:, :max_num_divided, :]  # First b * 64 * f part
                out2 = out[:, max_num_divided:, :]  # Second b * 64 * f part

                out1 = self.norm(out1)
                out2 = self.norm(out2)
                # out = self.norm(out) # this is the issue line
                out = torch.cat((out1,out2), dim=1)
                out[zero_mask] = 0
            else:
                print("there")
                out = self.norm(out)
        # print("after norm", out[zero_mask])
        if self.use_mlp:
            if self.pre_or_post == "pre":
                out_mlp = self.norm_mlp(out)
                out_mlp = self.mlp(out_mlp)

                if out.shape[-1] == out_mlp.shape[-1]:
                    out = out_mlp + out
            out[zero_mask] = 0

            if self.pre_or_post == "post":
                out_mlp = self.mlp(out)
                if out.shape[-1] == out_mlp.shape[-1]:
                    out = self.residual_mlp(out, out_mlp)

                b, max_num , f = out.shape
                # Split the tensor into two parts
                if hidden_adj_masked is not None:
                    max_num_divided = max_num//2
                    out1 = out[:, :max_num_divided, :]  # First b * 64 * f part
                    out2 = out[:, max_num_divided:, :]  # Second b * 64 * f part

                    out1 = self.norm_mlp(out1)
                    out2 = self.norm_mlp(out2)

                    # out = self.norm_mlp(out)
                    out = torch.cat((out1,out2), dim=1)
                    out[zero_mask] = 0
                else:
                    out = self.norm_mlp(out)
                    out[zero_mask] = 0
            # print("after post", out[zero_mask])
        if self.residual_dropout > 0:
            out = F.dropout(out, p=self.residual_dropout)
        # print("after ALL", out[zero_mask])
        return out, edge_index, batch_mapping, max_items, adj_mask, hidden_adj_masked, hidden_adj_self, zero_mask










class PMAComplete(nn.Module):
    def __init__(
        self,
        dim_hidden,
        num_heads,
        num_outputs,
        norm_type,
        dropout=0,
        use_mlp=False,
        mlp_hidden_size=64,
        mlp_type="standard",
        xformers_or_torch_attn="xformers",
        set_max_items=0,
        use_bfloat16=True,
        num_mlp_layers=3,
        pre_or_post="pre",
        num_layers_for_residual=0,
        residual_dropout=0,
        use_mlp_ln=False,
        mlp_dropout=0,
    ):
        super(PMAComplete, self).__init__()

        self.use_mlp = use_mlp
        self.mlp_hidden_size = mlp_hidden_size
        self.num_heads = num_heads
        self.xformers_or_torch_attn = xformers_or_torch_attn
        self.set_max_items = set_max_items
        self.use_bfloat16 = use_bfloat16
        self.residual_dropout = residual_dropout
        self.num_mlp_layers = num_mlp_layers
        self.pre_or_post = pre_or_post

        if self.pre_or_post == "post":
            self.residual_attn = admin_torch.as_module(num_layers_for_residual)
            self.residual_mlp = admin_torch.as_module(num_layers_for_residual)

        self.pma = PMA(dim_hidden, num_heads, num_outputs, dropout, xformers_or_torch_attn)

        if norm_type == "LN":
            print("LN", dim_hidden)
            self.norm = LN(dim_hidden)
        elif norm_type == "BN":
            print("BN", self.set_max_items)
            self.norm = BN(self.set_max_items)

        self.mlp_type = mlp_type

        if self.use_mlp:
            if self.mlp_type == "standard":
                self.mlp = SmallMLP(
                    in_dim=dim_hidden,
                    out_dim=dim_hidden,
                    inter_dim=mlp_hidden_size,
                    dropout_p=mlp_dropout,
                    num_layers=num_mlp_layers,
                    use_ln=use_mlp_ln,
                )

            elif self.mlp_type == "gated_mlp":
                self.mlp = GatedMLPMulti(
                    in_dim=dim_hidden,
                    out_dim=dim_hidden,
                    inter_dim=mlp_hidden_size,
                    dropout_p=mlp_dropout,
                    num_layers=num_mlp_layers,
                    use_ln=use_mlp_ln,
                )

        if norm_type == "LN":
            self.norm_mlp = LN(dim_hidden)
        elif norm_type == "BN":
            self.norm_mlp = BN(32)


    def forward(self, inp):
        X, edge_index, batch_mapping, max_items, adj_mask, hidden_adj_masked, hidden_adj_self, zero_mask = inp

        if self.pre_or_post == "pre":
            X = self.norm(X)

        out_attn = self.pma(X)

        if self.pre_or_post == "pre" and out_attn.shape[-2] == X.shape[-2]:
            out = X + out_attn
        
        elif self.pre_or_post == "post" and out_attn.shape[-2] == X.shape[-2]:
            out = self.residual_attn(X, out_attn)
            print("second there")
            print(out.shape, "out shape")
            out = self.norm(out)
        
        else:
            out = out_attn

        if self.use_mlp:
            if self.pre_or_post == "pre":
                out_mlp = self.norm_mlp(out)
                out_mlp = self.mlp(out_mlp)
                if out.shape[-2] == out_mlp.shape[-2]:
                    out = out_mlp + out

            if self.pre_or_post == "post":
                out_mlp = self.mlp(out)
                if out.shape[-2] == out_mlp.shape[-2]:
                    out = self.residual_mlp(out, out_mlp)
                out = self.norm_mlp(out)

        if self.residual_dropout > 0:
            out = F.dropout(out, p=self.residual_dropout)

        return out, edge_index, batch_mapping, max_items, adj_mask, hidden_adj_masked, hidden_adj_self, zero_mask



def zero_diag_batch(tensor, empty_mask_fill_value):
    # tensor: [B, n, n]
    B, n, _ = tensor.shape
    idx = torch.arange(n, device=tensor.device)
    tensor[:, idx, idx] = empty_mask_fill_value
    return tensor





def add_hidden_adj_masked(adj_mask_given, xformers_or_torch_attn, use_bfloat16, zero_mask):
    if xformers_or_torch_attn in ["torch"]:
        empty_mask_fill_value = False
        mask_dtype = torch.bool 
        edge_mask_fill_value = True
    else:
        empty_mask_fill_value = -99999
        mask_dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        edge_mask_fill_value = 0
    adj_mask = adj_mask_given.clone()
    B, _, n, _ = adj_mask.shape
    device = adj_mask.device
    # adj_and_hidden = torch.zeros((B, 1, 2*n, 2*n), dtype=torch.float32, device=device)
    adj_and_hidden = torch.full((B, 1, 2*n, 2*n), fill_value=empty_mask_fill_value, dtype=torch.float32, device=device)

    # Upper-left block
    upper_left = adj_mask[:, 0, :, :].clone()  # [B, n, n]
    upper_left = zero_diag_batch(upper_left, empty_mask_fill_value)
    adj_and_hidden[:, 0, :n, :n] = upper_left

    # Lower-left block
    lower_left = adj_mask[:, 0, :, :].clone()  # [B, n, n]
    lower_left = zero_diag_batch(lower_left, empty_mask_fill_value)
    adj_and_hidden[:, 0, n:, :n] = lower_left

    adj_and_hidden[zero_mask.unsqueeze(1)] = empty_mask_fill_value

    return adj_and_hidden



def add_hidden_adj_self(adj_mask_given, xformers_or_torch_attn, use_bfloat16, zero_mask):
    if xformers_or_torch_attn in ["torch"]:
        empty_mask_fill_value = False
        mask_dtype = torch.bool 
        edge_mask_fill_value = True
    else:
        empty_mask_fill_value = -99999
        mask_dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        edge_mask_fill_value = 0

    adj_mask = adj_mask_given.clone()
    B, _, n, _ = adj_mask.shape ; device = adj_mask.device
    # adj_and_hidden = torch.zeros((B, 1, 2*n, 2*n), dtype=torch.float32, device=device)
    adj_and_hidden = torch.full((B, 1, 2*n, 2*n), fill_value=empty_mask_fill_value, dtype=torch.float32, device=device)

    def ones_no_diag(B, n, device):# Helper to zero diagonal per batch
        mat = torch.full((B, n, n), fill_value=edge_mask_fill_value, dtype=torch.float32, device=device) # careful can still attend to padded rows...
        idx = torch.arange(n, device=device)
        mat[:, idx, idx] = empty_mask_fill_value
        return mat
    
    adj_and_hidden[:, 0, :n, :n] = ones_no_diag(B, n, device) # Upper-left block
    adj_and_hidden[:, 0, n:, :n] = ones_no_diag(B, n, device) # Lower-left block

    adj_and_hidden[zero_mask.unsqueeze(1)] = empty_mask_fill_value

    return adj_and_hidden
# issue when using torch likely















class ESA_hidden(nn.Module):
    def __init__(
        self,
        num_outputs,
        dim_output,
        dim_hidden,
        num_heads,
        layer_types,
        node_or_edge="edge",
        xformers_or_torch_attn="xformers",
        pre_or_post="pre",
        norm_type="LN",
        sab_dropout=0.0,
        mab_dropout=0.0,
        pma_dropout=0.0,
        residual_dropout=0.0,
        pma_residual_dropout=0.0,
        use_mlps=False,
        mlp_hidden_size=64,
        num_mlp_layers=2,
        mlp_type="gated_mlp",
        mlp_dropout=0.0,
        use_mlp_ln=False,
        set_max_items=0,
        use_bfloat16=True,
        reconstruction=False,
    ):
        super(ESA_hidden, self).__init__()

        assert len(layer_types) == len(dim_hidden) and len(layer_types) == len(num_heads)

        self.dim_hidden = dim_hidden
        self.num_heads = num_heads
        self.layer_types = layer_types
        self.node_or_edge = node_or_edge
        self.xformers_or_torch_attn = xformers_or_torch_attn
        self.pre_or_post = pre_or_post
        self.norm_type = norm_type
        self.sab_dropout = sab_dropout
        self.mab_dropout = mab_dropout
        self.pma_dropout = pma_dropout
        self.residual_dropout = residual_dropout
        self.pma_residual_dropout = pma_residual_dropout
        self.use_mlps = use_mlps
        self.mlp_hidden_size = mlp_hidden_size
        self.num_mlp_layers = num_mlp_layers
        self.mlp_type = mlp_type
        self.mlp_dropout = mlp_dropout
        self.use_mlp_ln = use_mlp_ln
        self.set_max_items = set_max_items
        print(self.set_max_items, "max items init")
        self.use_bfloat16 = use_bfloat16
        self.reconstruction = reconstruction
        
        layer_tracker = 0

        self.encoder = []

        pma_encountered = False
        dim_pma = -1

        has_pma = "P" in self.layer_types

        for lt in self.layer_types:
            layer_in_dim = dim_hidden[layer_tracker]
            layer_num_heads = num_heads[layer_tracker]
            if lt != "P":
                if has_pma:
                    layer_out_dim = dim_hidden[layer_tracker + 1]
                else:
                    layer_out_dim = dim_hidden[layer_tracker]
            else:
                layer_out_dim = -1

            if lt == "S" and not pma_encountered:
                self.encoder.append(
                    SABComplete(
                        layer_in_dim,
                        layer_out_dim,
                        layer_num_heads,
                        idx=0,
                        dropout=sab_dropout,
                        node_or_edge=node_or_edge,
                        xformers_or_torch_attn=xformers_or_torch_attn,
                        pre_or_post=pre_or_post,
                        residual_dropout=residual_dropout,
                        use_mlp=use_mlps,
                        mlp_hidden_size=mlp_hidden_size,
                        mlp_dropout=mlp_dropout,
                        num_mlp_layers=num_mlp_layers,
                        use_mlp_ln=use_mlp_ln,
                        norm_type=norm_type,
                        mlp_type=mlp_type,
                        set_max_items=set_max_items,
                        use_bfloat16=use_bfloat16,
                        num_layers_for_residual=len(dim_hidden) * 2,
                    )
                )
                
                print(f"Added encoder SAB ({layer_in_dim}, {layer_out_dim}, {layer_num_heads})")

            if lt == "M" and not pma_encountered:
                self.encoder.append(
                    SABComplete(
                        layer_in_dim,
                        layer_out_dim,
                        layer_num_heads,
                        idx=1,
                        dropout=mab_dropout,
                        node_or_edge=node_or_edge,
                        xformers_or_torch_attn=xformers_or_torch_attn,
                        pre_or_post=pre_or_post,
                        residual_dropout=residual_dropout,
                        use_mlp=use_mlps,
                        mlp_hidden_size=mlp_hidden_size,
                        mlp_dropout=mlp_dropout,
                        num_mlp_layers=num_mlp_layers,
                        use_mlp_ln=use_mlp_ln,
                        norm_type=norm_type,
                        mlp_type=mlp_type,
                        set_max_items=set_max_items,
                        use_bfloat16=use_bfloat16,
                        num_layers_for_residual=len(dim_hidden) * 2,
                    )
                )
                
                print(f"Added encoder MAB ({layer_in_dim}, {layer_out_dim}, {layer_num_heads})")
                
            if lt == "P":
                pma_encountered = True
                dim_pma = layer_in_dim
                self.decoder = [
                    PMAComplete(
                        layer_in_dim,
                        layer_num_heads,
                        num_outputs,
                        dropout=pma_dropout,
                        residual_dropout=pma_residual_dropout,
                        xformers_or_torch_attn=xformers_or_torch_attn,
                        pre_or_post=pre_or_post,
                        use_mlp=use_mlps,
                        mlp_hidden_size=mlp_hidden_size,
                        mlp_dropout=mlp_dropout,
                        num_mlp_layers=num_mlp_layers,
                        use_mlp_ln=use_mlp_ln,
                        norm_type=norm_type,
                        mlp_type=mlp_type,
                        set_max_items=set_max_items,
                        use_bfloat16=use_bfloat16,
                        num_layers_for_residual=len(dim_hidden) * 2,
                    )
                ]

                print(f"Added decoder PMA ({layer_in_dim}, {layer_num_heads})")

            if lt == "S" and pma_encountered:
                self.decoder.append(
                    SABComplete(
                        layer_in_dim,
                        layer_out_dim,
                        layer_num_heads,
                        idx=2,
                        dropout=sab_dropout,
                        node_or_edge=node_or_edge,
                        xformers_or_torch_attn=xformers_or_torch_attn,
                        pre_or_post=pre_or_post,
                        residual_dropout=residual_dropout,
                        use_mlp=use_mlps,
                        mlp_hidden_size=mlp_hidden_size,
                        mlp_dropout=mlp_dropout,
                        num_mlp_layers=num_mlp_layers,
                        use_mlp_ln=use_mlp_ln,
                        norm_type=norm_type,
                        mlp_type=mlp_type,
                        set_max_items=set_max_items,
                        use_bfloat16=use_bfloat16,
                        num_layers_for_residual=len(dim_hidden) * 2,
                    )
                )

                print(f"Added decoder SAB ({layer_in_dim}, {layer_out_dim}, {layer_num_heads})")

            if lt != "P":
                layer_tracker += 1

        self.encoder = nn.Sequential(*self.encoder)
        if pma_encountered:
            self.decoder = nn.Sequential(*self.decoder)

        self.decoder_linear = nn.Linear(dim_hidden[-1], dim_output, bias=True)

        if has_pma and dim_hidden[0] != dim_pma:
            self.out_proj = nn.Linear(dim_hidden[0], dim_pma)

            self.dim_pma = dim_pma

        if self.reconstruction:
            print("ready to reconstruct")

            reversed_layer_types = self.layer_types[::-1]
            reversed_dims = self.dim_hidden[::-1]
            reversed_heads = self.num_heads[::-1]

            layer_tracker = 0
            self.reconstructor = []

            for i, lt in enumerate(reversed_layer_types):
                layer_in_dim = reversed_dims[layer_tracker]
                if i + 1 < len(reversed_dims):
                    layer_out_dim = reversed_dims[layer_tracker + 1]
                else:
                    layer_out_dim = reversed_dims[layer_tracker]

                layer_num_heads = reversed_heads[layer_tracker]



                if lt == "S" or lt == "M":
                    self.reconstructor.append(
                        SABComplete(
                            layer_in_dim,
                            layer_out_dim,
                            layer_num_heads, 
                            idx=999 - i,  # optional: to distinguish from encoder
                            dropout=self.sab_dropout if lt == "S" else self.mab_dropout,
                            node_or_edge=self.node_or_edge,
                            xformers_or_torch_attn=self.xformers_or_torch_attn,
                            pre_or_post=self.pre_or_post,
                            residual_dropout=self.residual_dropout,
                            use_mlp=self.use_mlps,
                            mlp_hidden_size=self.mlp_hidden_size,
                            mlp_dropout=self.mlp_dropout,
                            num_mlp_layers=self.num_mlp_layers,
                            use_mlp_ln=self.use_mlp_ln,
                            norm_type=self.norm_type,
                            mlp_type=self.mlp_type,
                            set_max_items=self.set_max_items,
                            use_bfloat16=self.use_bfloat16,
                            num_layers_for_residual=len(self.dim_hidden) * 2,
                        )
                    )
                    print(f"Added reconstructor {lt} ({layer_in_dim}, {layer_out_dim}, {layer_num_heads})")
                    
                if lt == "P":
                    print("PMA ENCOUNTERED !!!!!!!")
                layer_tracker += 1

            self.reconstructor = nn.Sequential(*self.reconstructor)


    def forward(self, X, edge_index, batch_mapping, num_max_items, is_using_hyperedges=False, hyperedge_index=None, hedge_batch_index=None, zero_mask=None):

        # modify this to be the perfect adj_mask
        if self.node_or_edge == "edge":
            adj_mask = get_adj_mask_from_edge_index_edge(
                edge_index=edge_index,
                batch_mapping=batch_mapping,
                batch_size=X.shape[0],  
                max_items=self.set_max_items,
                xformers_or_torch_attn=self.xformers_or_torch_attn,
                use_bfloat16=self.use_bfloat16,
                is_using_hyperedges=is_using_hyperedges,
                hyperedge_index=hyperedge_index,
                hedge_batch_index=hedge_batch_index,
            )
            hidden_adj_masked = None; hidden_adj_self = None
            if zero_mask is not None:
                hidden_adj_masked = add_hidden_adj_masked(adj_mask, xformers_or_torch_attn=self.xformers_or_torch_attn, use_bfloat16=self.use_bfloat16, zero_mask = zero_mask)
                hidden_adj_self = add_hidden_adj_self(adj_mask, xformers_or_torch_attn=self.xformers_or_torch_attn, use_bfloat16=self.use_bfloat16, zero_mask = zero_mask)

        # elif self.node_or_edge == "node":
        #     adj_mask = get_adj_mask_from_edge_index_node(
        #         edge_index=edge_index,
        #         batch_mapping=batch_mapping,
        #         batch_size=X.shape[0],
        #         max_items=self.set_max_items,
        #         xformers_or_torch_attn=self.xformers_or_torch_attn,
        #         use_bfloat16=self.use_bfloat16,
        #     )
        
        # issue is when is_using_hyperedges, adj_mask contains the hyperedges connectivity, but the edge_index and the batch_mapping don't
        # batch_mapping links node with graphs, so not an issue
        # # # the true issue is edge_index which is only edges, not hyperedges
        if hidden_adj_masked is not None and hidden_adj_self is not None :
            enc, _, _, _, _, _, _, _ = self.encoder((X, edge_index, batch_mapping, num_max_items, adj_mask, hidden_adj_masked, hidden_adj_self, zero_mask))
        else:
            enc, _, _, _, _, _, _, _ = self.encoder((X, edge_index, batch_mapping, num_max_items, adj_mask, None, None, zero_mask))

        if hasattr(self, "dim_pma") and self.dim_hidden[0] != self.dim_pma:
            X = self.out_proj(X)
        enc = enc + X
        enc[zero_mask] = 0

        # print(enc.shape)
        
        _, max_num_enc, _ = enc.shape ; max_num_enc_div = max_num_enc//2
        enc_normal = enc[:, :max_num_enc_div, :] ; enc_hidden = enc[:, max_num_enc_div:, :]  


        ## -- Observatory --
        l2_loss, l2_loss_normal, l2_loss_hidden, mean_rank, mean_hidden_rank = 0,0,0,0,0
        #with torch.no_grad():
            # normal_ranks = batched_effective_rank(enc_normal) ; hidden_ranks = batched_effective_rank(enc_hidden)
            # mean_rank = normal_ranks.float().mean().detach() ; mean_hidden_rank = hidden_ranks.mean().detach()

            # mean_enc = enc.mean(dim=1, keepdim=True) ; dist_l2 = ((enc - mean_enc) ** 2).sum(dim=2).sqrt() ; l2_loss = dist_l2.mean().detach()
            # mean_enc_normal = enc_normal.mean(dim=1, keepdim=True) ; dist_l2_normal = ((enc_normal - mean_enc_normal) ** 2).sum(dim=2).sqrt() ; l2_loss_normal = dist_l2_normal.mean().detach()
            # mean_enc_hidden = enc_hidden.mean(dim=1, keepdim=True) ; dist_l2_hidden = ((enc_hidden - mean_enc_hidden) ** 2).sum(dim=2).sqrt() ; l2_loss_hidden = dist_l2_hidden.mean().detach()

        #     Dist_wasserstein_normal = compute_sinkhorn_distance_matrix(enc_normal)
        #     Dist_wasserstein_hidden = compute_sinkhorn_distance_matrix(enc_hidden)
            # cka_score = linear_CKA(Dist_wasserstein_normal, Dist_wasserstein_hidden)
        cka_score = 0
        ## ----------------

        enc_target = (enc_normal.detach()).clone()
        latent_rep_loss = F.mse_loss(enc_hidden, enc_target, reduction='mean')   # here we compare enc_normal and enc_hidden

        if hasattr(self, "decoder"):
            # out, _, _, _, _, _, _ = self.decoder((enc, edge_index, batch_mapping, num_max_items, adj_mask, hidden_adj_masked, hidden_adj_self))
            out, _, _, _, _, _, _, _ = self.decoder((enc_normal, edge_index, batch_mapping, num_max_items, adj_mask, None, None, zero_mask))
            out = out.mean(dim=1)
        else:
            out = enc_normal

        if hasattr(self, "reconstructor"):
            # goes from enc.shape to 
            # print("going to reconstruct")
            enc_for_reconstruction = enc.detach().clone()
            enc_for_reconstruction[zero_mask] = 0
            reconstructed, _, _, _, _, _, _, _ = self.reconstructor((enc_for_reconstruction, edge_index, batch_mapping, num_max_items, adj_mask, hidden_adj_masked, hidden_adj_self, zero_mask))
            reconstructed[zero_mask] = 0
        else:
            print("no reconstructor...")
            reconstructed=None

        return F.mish(self.decoder_linear(out)), latent_rep_loss, reconstructed, (l2_loss, l2_loss_normal, l2_loss_hidden, mean_rank, mean_hidden_rank, cka_score)
     