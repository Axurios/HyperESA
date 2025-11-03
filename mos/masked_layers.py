import torch
import admin_torch
import math
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import unbatch_edge_index

from utils.norm_layers import BN, LN
from mos.mha import SAB, PMA, S_2Simp_AB, S_3Simp_AB
from mos.mlp_utils import SmallMLP, GatedMLPMulti


# def parse_layers(tokens, i=0):
#     result = []
#     while i < len(tokens):
#         tok = tokens[i]

#         if tok == "R":  # router starts: expect '['
#             assert tokens[i+1] == "[", f"Expected '[', got {tokens[i+1]}"
#             inner, j = parse_layers(tokens, i+2)  # parse inside router
#             result.append({"R": inner})
#             i = j
#         elif tok == "]":  # router end
#             return result, i + 1
#         else:
#             result.append(tok)
#             i += 1
#     return result
def parse_layers(tokens, i=0):
    result = []
    while i < len(tokens):
        tok = tokens[i]
        if tok == "R" or tok == "L" or tok=="T":  # Either router or layer starts: expect '['
            assert tokens[i + 1] == "[", f"Expected '[', got {tokens[i + 1]}"
            inner, j = parse_layers(tokens, i + 2)  # parse inside the router or layer
            
            if tok == "R":
                result.append({"R": inner})  # Add router
            elif tok == "L":
                result.append({"L": inner})  # Add layer
            elif tok == "T":
                result.append({"T": inner})  # Add layer
            i = j
            
        elif tok == "]":  # Router or Layer end
            return result, i + 1
        else:
            result.append(tok) ; i += 1
    return result




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
    # Calculate the length of each segment
    lengths = input_tensor[1:] - input_tensor[:-1]

    # Append the final length
    lengths = torch.cat((lengths, torch.tensor([final - input_tensor[-1]], device=torch.device("cuda:0"))))

    # Create ranges for each segment
    ranges = [torch.arange(0, length, device=torch.device("cuda:0")) for length in lengths]

    # Concatenate all ranges into a single tensor
    result = torch.cat(ranges)

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
    
    # print(max_items, "max items")
    # print(edge_index.shape[1], "edge index shape")
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
        
        if self.idx == 0 or self.idx == 1:
            self.sab = SAB(dim_in, dim_out, num_heads, dropout, xformers_or_torch_attn)


        if self.idx == 3:
            print("2-simplicial attention layer")
            self.sab = S_2Simp_AB(dim_in, dim_out, num_heads, dropout, xformers_or_torch_attn)

        if self.idx == 4:
            print("3-simplicial attention layer")
            self.sab = S_3Simp_AB(dim_in, dim_out, num_heads, dropout, xformers_or_torch_attn)


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
        X, edge_index, batch_mapping, max_items, adj_mask = inp

        if self.pre_or_post == "pre":
            X = self.norm(X) # also work when hyperedges in x
        if self.idx == 1:
            out_attn = self.sab(X, adj_mask)
        else:
            out_attn = self.sab(X, None)
        
        if out_attn.shape[-1] != X.shape[-1]:
            X = self.proj_1(X)
        
        if self.pre_or_post == "pre":
            out = X + out_attn
        
        if self.pre_or_post == "post":
            out = self.residual_attn(X, out_attn)
            out = self.norm(out)
        
        if self.use_mlp:
            if self.pre_or_post == "pre":
                out_mlp = self.norm_mlp(out)
                out_mlp = self.mlp(out_mlp)
                if out.shape[-1] == out_mlp.shape[-1]:
                    out = out_mlp + out

            if self.pre_or_post == "post":
                out_mlp = self.mlp(out)
                if out.shape[-1] == out_mlp.shape[-1]:
                    out = self.residual_mlp(out, out_mlp)
                out = self.norm_mlp(out)

        if self.residual_dropout > 0:
            out = F.dropout(out, p=self.residual_dropout)
        return out, edge_index, batch_mapping, max_items, adj_mask


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
            self.norm = LN(dim_hidden)
        elif norm_type == "BN":
            # print(self.set_max_items, "here max items is")
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
        X, edge_index, batch_mapping, max_items, adj_mask = inp

        if self.pre_or_post == "pre":
            X = self.norm(X)

        out_attn = self.pma(X)

        if self.pre_or_post == "pre" and out_attn.shape[-2] == X.shape[-2]:
            out = X + out_attn
        
        elif self.pre_or_post == "post" and out_attn.shape[-2] == X.shape[-2]:
            out = self.residual_attn(X, out_attn)
            # print(out.shape, "out shape before issue norm")
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

        return out, edge_index, batch_mapping, max_items, adj_mask












class Router(nn.Module):
    def __init__(
        self,
        layers_inside,
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
        activation="sigmoid",
        sab_dropout=0,
        mab_dropout=0,
    ):
        super(Router, self).__init__()
        # want seq_len to get k, want k to get 

        # self.dim_in = dim_in ; self.dim_out = dim_out
        self.linear = nn.Linear(dim_in, 1, bias=False)
        self.activation = activation

        self.layers_in = layers_inside
        print(layers_inside)
        
        self.router_seq = []
        # need new set_max_items,
        # print(set_max_items)
        self.k = set_max_items # if just attention
        # self.k = int(set_max_items ** (2/3))
        # self.k3 = int(set_max_items ** 0.5)
        # print(self.k, "nbr of tokens left k")
        # print(self.k3, "k3")
        if "M2" in self.layers_in:
            self.k = self.k = int(set_max_items ** (2/3))+1
        if "M3" in self.layers_in:
            self.k = int(set_max_items ** 0.5)+1

        for lt in self.layers_in:
            if lt in ["S", "M", "M2", "M3"]:
                idx_mapping = {"S": 0, "M": 1, "M2": 3, "M3": 4}
                idx = idx_mapping.get(lt)

                self.router_seq.append(
                    SABComplete(  
                        dim_in,
                        dim_out,
                        num_heads,
                        idx=idx,
                        dropout=dropout,
                        node_or_edge=node_or_edge,
                        xformers_or_torch_attn=xformers_or_torch_attn,
                        pre_or_post=pre_or_post,
                        residual_dropout=residual_dropout,
                        use_mlp=use_mlp,
                        mlp_hidden_size=mlp_hidden_size,
                        mlp_dropout=mlp_dropout,
                        num_mlp_layers=num_mlp_layers,
                        use_mlp_ln=use_mlp_ln,
                        norm_type=norm_type,
                        mlp_type=mlp_type,
                        set_max_items=self.k,
                        use_bfloat16=use_bfloat16,
                        num_layers_for_residual=num_layers_for_residual,
                    )
                )
                print(f"Added encoder {lt} ", f"({dim_in}, {dim_out}, {num_heads})")
                

            if isinstance(lt, dict) and "R" in lt:
                # print("found a dict", lt)
                idx = 0 if lt == "S" else 1
                dropout_val = sab_dropout if lt == "S" else mab_dropout
                self.router_seq.append(
                    Router(
                        lt['R'],
                        dim_in,
                        dim_out,
                        num_heads,
                        idx=idx,
                        dropout=dropout_val,
                        node_or_edge=node_or_edge,
                        xformers_or_torch_attn=xformers_or_torch_attn,
                        pre_or_post=pre_or_post,
                        residual_dropout=residual_dropout,
                        use_mlp=use_mlp,
                        mlp_hidden_size=mlp_hidden_size,
                        mlp_dropout=mlp_dropout,
                        num_mlp_layers=num_mlp_layers,
                        use_mlp_ln=use_mlp_ln,
                        norm_type=norm_type,
                        mlp_type=mlp_type,
                        set_max_items=self.k,
                        use_bfloat16=use_bfloat16,
                        num_layers_for_residual=num_layers_for_residual,
                    )
                )
                print(f"Added router ({dim_in}, {dim_out}, {num_heads}) with {lt['R']} inside")
        self.router_seq = nn.Sequential(*self.router_seq)
                

    def forward(self, inp):
        # print("in router forward")
        X, edge_index, batch_mapping, max_items, adj_mask = inp

        scores = self.linear(X).squeeze(-1)  # [batch, seq_len]
        
        if self.activation == "sigmoid":
            scores = torch.sigmoid(scores)
        elif self.activation == "tanh":
            scores = torch.tanh(scores)
        else:
            raise NotImplementedError
        
        batch_size, seq_len, hidden_dim = X.shape

        X_clone = X.clone()

        topk_scores, topk_indices = torch.topk(scores, self.k, dim=1)  # [batch, k] # top-k indices along seq_len dimension for each batch
        X_topk = torch.gather(
            X, 1, topk_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
        )  # [batch, k, hidden_dim]
        # print(X_topk.shape)

        enc_Xtopk, _, _, _, _ = self.router_seq((X_topk, edge_index, batch_mapping, self.k, None)) # this updates X_topk
        # X_topk, edge_index, batch_mapping, max_items, adj_mask = self.router_seq((X_topk, edge_index, batch_mapping, max_items, adj_mask))

        topk_scores_reshaped = topk_scores.unsqueeze(-1)  # [batch, k, 1]
        X_topk_weighted = enc_Xtopk * topk_scores_reshaped

        out = X_clone.scatter_add_(1, topk_indices.unsqueeze(-1).expand(-1, -1, hidden_dim), X_topk_weighted)  # print(X.shape)
        return out, edge_index, batch_mapping, max_items, adj_mask










class RecursiveRouter(nn.Module):
    def __init__(
        self,
        layers_inside,
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
        activation="sigmoid",
        sab_dropout=0,
        mab_dropout=0,
        recursive_depth=3,
    ):
        super(RecursiveRouter, self).__init__()

        self.recursive_depth = recursive_depth
        self.linear = nn.Linear(dim_in, 1, bias=False)
        self.activation = activation

        self.layers_in = layers_inside
        # print(layers_inside)
        
        self.router_seq = []
        self.k = set_max_items # if just attention

        for lt in self.layers_in:
            if lt in ["S", "M", "M2", "M3"]:
                idx_mapping = {"S": 0, "M": 1, "M2": 3, "M3": 4}
                idx = idx_mapping.get(lt)

                self.router_seq.append(
                    SABComplete(  
                        dim_in,
                        dim_out,
                        num_heads,
                        idx=idx,
                        dropout=dropout,
                        node_or_edge=node_or_edge,
                        xformers_or_torch_attn=xformers_or_torch_attn,
                        pre_or_post=pre_or_post,
                        residual_dropout=residual_dropout,
                        use_mlp=use_mlp,
                        mlp_hidden_size=mlp_hidden_size,
                        mlp_dropout=mlp_dropout,
                        num_mlp_layers=num_mlp_layers,
                        use_mlp_ln=use_mlp_ln,
                        norm_type=norm_type,
                        mlp_type=mlp_type,
                        set_max_items=self.k,
                        use_bfloat16=use_bfloat16,
                        num_layers_for_residual=num_layers_for_residual,
                    )
                )
                print(f"Added recursive encoder {lt} ", f"({dim_in}, {dim_out}, {num_heads})")
        self.router_seq = nn.Sequential(*self.router_seq)
                

    def forward(self, inp):
        X, edge_index, batch_mapping, max_items, adj_mask = inp
        scores = self.linear(X).squeeze(-1)  # [batch, seq_len]
        
        if self.activation == "sigmoid":
            scores = torch.sigmoid(scores)
        elif self.activation == "tanh":
            scores = torch.tanh(scores)
        else:
            raise NotImplementedError
        
        batch_size, seq_len, hidden_dim = X.shape
        X_clone = X.clone()

        topk_scores, topk_indices = torch.topk(scores, self.k, dim=1)  # [batch, k] # top-k indices along seq_len dimension for each batch
        X_topk = torch.gather(
            X, 1, topk_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
        )  # [batch, k, hidden_dim]


        for depth in range(self.recursive_depth):
            # print(f"Processing recursive depth {depth + 1} of {self.recursive_depth}")
            # Pass selected tokens through the corresponding router layer
            enc_Xtopk, _, _, _, _ = self.router_seq((X_topk, edge_index, batch_mapping, self.k, None))

            # Shrink the number of selected tokens by half each time
            next_k = self.k // 2 if self.k > 1 else 1  # a bit arbitrary check MoR

            # Update X_topk with the new selected tokens
            topk_scores, topk_indices = torch.topk(scores, self.k, dim=1)
            X_topk = torch.gather(
                X, 1, topk_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
            )  # [batch, next_k, hidden_dim]

        # After all recursive layers, gather the final output
        topk_scores_reshaped = topk_scores.unsqueeze(-1)  # [batch, k, 1]
        X_topk_weighted = enc_Xtopk * topk_scores_reshaped

        out = X_clone.scatter_add_(1, topk_indices.unsqueeze(-1).expand(-1, -1, hidden_dim), X_topk_weighted)
        return out, edge_index, batch_mapping, max_items, adj_mask
    

        # enc_Xtopk, _, _, _, _ = self.router_seq((X_topk, edge_index, batch_mapping, self.k, None)) # this updates X_topk
        # # X_topk, edge_index, batch_mapping, max_items, adj_mask = self.router_seq((X_topk, edge_index, batch_mapping, max_items, adj_mask))

        # topk_scores_reshaped = topk_scores.unsqueeze(-1)  # [batch, k, 1]
        # X_topk_weighted = enc_Xtopk * topk_scores_reshaped

        # out = X_clone.scatter_add_(1, topk_indices.unsqueeze(-1).expand(-1, -1, hidden_dim), X_topk_weighted)  # print(X.shape)
        # return out, edge_index, batch_mapping, max_items, adj_mask







class TRMRouter(nn.Module):
    def __init__(
        self,
        layers_inside,
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
        activation="sigmoid",
        sab_dropout=0,
        mab_dropout=0,
        recursive_depth=3,
    ):
        super(RecursiveRouter, self).__init__()

        self.recursive_depth = recursive_depth
        # self.linear = nn.Linear(dim_in, 1, bias=False)
        self.activation = activation

        self.layers_in = layers_inside
        # print(layers_inside)
        
        self.router_seq = []
        # self.k = set_max_items # if just attention

        for lt in self.layers_in:
            if lt in ["S", "M", "M2", "M3"]:
                idx_mapping = {"S": 0, "M": 1, "M2": 3, "M3": 4}
                idx = idx_mapping.get(lt)

                self.router_seq.append(
                    SABComplete(  
                        dim_in,
                        dim_out,
                        num_heads,
                        idx=idx,
                        dropout=dropout,
                        node_or_edge=node_or_edge,
                        xformers_or_torch_attn=xformers_or_torch_attn,
                        pre_or_post=pre_or_post,
                        residual_dropout=residual_dropout,
                        use_mlp=use_mlp,
                        mlp_hidden_size=mlp_hidden_size,
                        mlp_dropout=mlp_dropout,
                        num_mlp_layers=num_mlp_layers,
                        use_mlp_ln=use_mlp_ln,
                        norm_type=norm_type,
                        mlp_type=mlp_type,
                        set_max_items=self.k,
                        use_bfloat16=use_bfloat16,
                        num_layers_for_residual=num_layers_for_residual,
                    )
                )
                print(f"Added recursive encoder {lt} ", f"({dim_in}, {dim_out}, {num_heads})")
        self.router_seq = nn.Sequential(*self.router_seq)
                

    def forward(self, inp):
        X, edge_index, batch_mapping, max_items, adj_mask = inp
        batch_size, seq_len, hidden_dim = X.shape

        out = X

        for depth in range(self.recursive_depth - 1):
            with torch.no_grad():
                out, _, _, _, _ = self.router_seq((out, edge_index, batch_mapping, max_items, adj_mask))
        # Final recursion step with gradients
        out, _, _, _, _ = self.router_seq((out, edge_index, batch_mapping, max_items, adj_mask))

        return out, edge_index, batch_mapping, max_items, adj_mask
    
























class ESA(nn.Module):
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
    ):
        super(ESA, self).__init__()

        
        

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
        self.use_bfloat16 = use_bfloat16
        
        layer_tracker = 0
        # print(self.layer_types)
        self.layer_parsed = parse_layers(self.layer_types)
        print(self.layer_parsed)
        assert len(self.layer_parsed) == len(dim_hidden) and len(self.layer_parsed) == len(num_heads)
        self.encoder = []

        pma_encountered = False
        dim_pma = -1

        has_pma = "P" in self.layer_parsed

        for lt in self.layer_parsed:
            layer_in_dim = dim_hidden[layer_tracker]
            layer_num_heads = num_heads[layer_tracker]
            if lt != "P":
                if has_pma:
                    layer_out_dim = dim_hidden[layer_tracker + 1]
                else:
                    layer_out_dim = dim_hidden[layer_tracker]
            else:
                layer_out_dim = -1
            
            if lt in ["S", "M", "M2", "M3"] and not pma_encountered:
                # idx = 0 if lt == "S" else 1
                idx_mapping = {"S": 0, "M": 1, "M2": 3, "M3": 4}
                idx = idx_mapping.get(lt)
                dropout_val = sab_dropout if lt == "S" else mab_dropout

                self.encoder.append(
                    SABComplete(
                        layer_in_dim,
                        layer_out_dim,
                        layer_num_heads,
                        idx=idx,
                        dropout=dropout_val,
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
                print(f"Added encoder {'SAB' if lt == 'S' else 'MAB'} ", f"({layer_in_dim}, {layer_out_dim}, {layer_num_heads})")
                
            router_types = {
                "R": Router,
                "L": RecursiveRouter,
                "T": RecursiveRouter  # if L and T both map to RecursiveRouter
            }
            max_recursive_depth = 3
            # max_recursive_depth = hyperparams.get("recursive_depth", 3)   
            if isinstance(lt, dict):
                for key, router_class in router_types.items():
                    if key in lt:
                        layer_config = lt[key]
                        self.encoder.append(
                            router_class(
                                layer_config,
                                layer_in_dim,
                                layer_out_dim,
                                layer_num_heads,
                                idx=idx,
                                dropout=dropout_val,
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
                                sab_dropout=sab_dropout,
                                mab_dropout=mab_dropout,
                                **({"recursive_depth": max_recursive_depth} if key in ["L", "T"] else {})
                            )
                        )
                        print(f"Added {'TRM' if key == 'T' else 'Recursive' if key == 'L' else 'Router'} "
                            f"({layer_in_dim}, {layer_out_dim}, {layer_num_heads}) with {layer_config} inside")
            # if isinstance(lt, dict) and "R" in lt:
            #     # print("found a dict", lt)
            #     # idx = 0 if lt == "S" else 1
            #     # dropout_val = sab_dropout if lt == "S" else mab_dropout

            #     self.encoder.append(
            #         Router(
            #             lt['R'],
            #             layer_in_dim,
            #             layer_out_dim,
            #             layer_num_heads,
            #             idx=idx,
            #             dropout=dropout_val,
            #             node_or_edge=node_or_edge,
            #             xformers_or_torch_attn=xformers_or_torch_attn,
            #             pre_or_post=pre_or_post,
            #             residual_dropout=residual_dropout,
            #             use_mlp=use_mlps,
            #             mlp_hidden_size=mlp_hidden_size,
            #             mlp_dropout=mlp_dropout,
            #             num_mlp_layers=num_mlp_layers,
            #             use_mlp_ln=use_mlp_ln,
            #             norm_type=norm_type,
            #             mlp_type=mlp_type,
            #             set_max_items=set_max_items,
            #             use_bfloat16=use_bfloat16,
            #             num_layers_for_residual=len(dim_hidden) * 2,
            #             sab_dropout = sab_dropout,
            #             mab_dropout= mab_dropout,
            #         )
            #     )
            #     print(f"Added Router ({layer_in_dim}, {layer_out_dim}, {layer_num_heads}) with {lt['R']} inside")
            
            # if isinstance(lt, dict) and "L" in lt:
            #     # how to put max recursive depth in hyperparameters ?
            #     max_recursive_depth = 3
            #     self.encoder.append(
            #         RecursiveRouter(
            #             lt['L'],
            #             layer_in_dim,
            #             layer_out_dim,
            #             layer_num_heads,
            #             idx=idx,
            #             dropout=dropout_val,
            #             node_or_edge=node_or_edge,
            #             xformers_or_torch_attn=xformers_or_torch_attn,
            #             pre_or_post=pre_or_post,
            #             residual_dropout=residual_dropout,
            #             use_mlp=use_mlps,
            #             mlp_hidden_size=mlp_hidden_size,
            #             mlp_dropout=mlp_dropout,
            #             num_mlp_layers=num_mlp_layers,
            #             use_mlp_ln=use_mlp_ln,
            #             norm_type=norm_type,
            #             mlp_type=mlp_type,
            #             set_max_items=set_max_items,
            #             use_bfloat16=use_bfloat16,
            #             num_layers_for_residual=len(dim_hidden) * 2,
            #             sab_dropout = sab_dropout,
            #             mab_dropout= mab_dropout,
            #             recursive_depth=max_recursive_depth,  # hardcoded for now
            #         )
            #     )
            #     print(f"Added Recursive router (loop) ({layer_in_dim}, {layer_out_dim}, {layer_num_heads}) with {lt['L']} inside")
            
            # if isinstance(lt, dict) and "T" in lt:
            #     # how to put max recursive depth in hyperparameters ?
            #     max_recursive_depth = 3
            #     self.encoder.append(
            #         RecursiveRouter(
            #             lt['T'],
            #             layer_in_dim,
            #             layer_out_dim,
            #             layer_num_heads,
            #             idx=idx,
            #             dropout=dropout_val,
            #             node_or_edge=node_or_edge,
            #             xformers_or_torch_attn=xformers_or_torch_attn,
            #             pre_or_post=pre_or_post,
            #             residual_dropout=residual_dropout,
            #             use_mlp=use_mlps,
            #             mlp_hidden_size=mlp_hidden_size,
            #             mlp_dropout=mlp_dropout,
            #             num_mlp_layers=num_mlp_layers,
            #             use_mlp_ln=use_mlp_ln,
            #             norm_type=norm_type,
            #             mlp_type=mlp_type,
            #             set_max_items=set_max_items,
            #             use_bfloat16=use_bfloat16,
            #             num_layers_for_residual=len(dim_hidden) * 2,
            #             sab_dropout = sab_dropout,
            #             mab_dropout= mab_dropout,
            #             recursive_depth=max_recursive_depth,  # hardcoded for now
            #         )
            #     )
            #     print(f"Added TRM router (loop) ({layer_in_dim}, {layer_out_dim}, {layer_num_heads}) with {lt['T']} inside")
                
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


    def forward(self, X, edge_index, batch_mapping, num_max_items, is_using_hyperedges=False, hyperedge_index=None, hedge_batch_index=None,):
        norms_list = [] ; rank_list = []
        if self.node_or_edge == "node":
            adj_mask = get_adj_mask_from_edge_index_node(
                edge_index=edge_index,
                batch_mapping=batch_mapping,
                batch_size=X.shape[0],
                max_items=self.set_max_items,
                xformers_or_torch_attn=self.xformers_or_torch_attn,
                use_bfloat16=self.use_bfloat16,
            )
        elif self.node_or_edge == "edge":
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
        
        # issue is when is_using_hyperedges, adj_mask contains the hyperedges connectivity, but the edge_index and the batch_mapping don't
        # batch_mapping links node with graphs, so not an issue
        # the true issue is edge_index which is only edges, not hyperedges
        # print("okok")
        # enc, _, _, _, _ = self.encoder((X, edge_index, batch_mapping, num_max_items, adj_mask))
        enc = X.clone()
        for layer_idx in range(len(self.encoder)):
            layer = self.encoder[layer_idx]  # get the layer from nn.Sequential
            # if layer is a router, then do smthing to keep the x, etc.
            enc, _, _,  _, _ = layer((enc, edge_index, batch_mapping, num_max_items, adj_mask))  # forward pass through the layer
            with torch.no_grad():
                mu_norm = compute_mu_norm(enc)
                #norms_list.append(mu_norm)

                #eff_rank = batched_effective_rank(enc)
                #rank_list.append(eff_rank)

        if hasattr(self, "dim_pma") and self.dim_hidden[0] != self.dim_pma:
            X = self.out_proj(X)

        enc = enc + X
        # print("enc shape", enc.shape)
        #norms_tensor = torch.stack(norms_list, dim=0)
        #rank_tensor = torch.stack(rank_list, dim=0)
        # print(norms_tensor.shape, "norms tensor shape")
        # latent_rep_loss = 0.0 

        if hasattr(self, "decoder"):
            out, _, _, _, _ = self.decoder((enc, edge_index, batch_mapping, num_max_items, adj_mask))
            out = out.mean(dim=1)
        else:
            out = enc

        #return F.mish(self.decoder_linear(out)), enc, norms_tensor, rank_tensor
        return F.mish(self.decoder_linear(out)), enc
    




def compute_mu_norm(X):
    # print(X.shape)
    batch_size, n, d = X.shape
    # Step 1: Compute γX = (1^T X) / N # Sum along the rows (dim=0), then divide by N
    # gamma_X = X.sum(dim=0, keepdim=True) / N
    gamma_X = X.sum(dim=(1, 2), keepdim=True) / (n * d)
    # Step 2: Compute the residual X - γX
    residual = X - gamma_X
    # Step 3: Compute the Frobenius norm of the residual
    frobenius_norm = torch.norm(residual, p='fro', dim=(1, 2))  # Frobenius norm
    original_frobenius_norm = torch.norm(X, p='fro', dim=(1, 2))
    relative_norm = frobenius_norm / original_frobenius_norm
    # print(relative_norm.shape, "frob norm")
    return relative_norm



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