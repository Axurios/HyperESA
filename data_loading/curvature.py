import numpy as np
import pandas as pd
import os
import torch
import torch_geometric
from torch_geometric.data import Data
import random as rd
# from torch_geometric.utils import degree, dense_to_sparse
# from torch_geometric.data import Data, InMemoryDataset
from tqdm.auto import tqdm
import matplotlib.cm as cm
import seaborn as sns


def _unique_undirected_edge_list(edge_index, num_nodes):
    """
    Return sorted list of unique undirected edges as tensor [2, E] with u < v.
    100% portable: no `return_index` usage.
    """
    ei = edge_index.clone()
    if ei.ndim != 2 or ei.shape[0] != 2:
        raise ValueError("edge_index must be shape [2, M]")

    # ensure (min,max) per edge
    mins = torch.min(ei, dim=0).values
    maxs = torch.max(ei, dim=0).values
    und = torch.stack([mins, maxs], dim=0)
    mask = und[0] != und[1]
    und = und[:, mask]
    if und.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)

    # compute unique keys (u,v)
    keys = und[0].to(torch.int64) * (num_nodes + 1) + und[1].to(torch.int64)
    unique_keys = torch.unique(keys, sorted=True)
    # map back: keep only the first matching occurrence of each key
    _, first_indices = torch.tensor([], dtype=torch.long), []
    seen = set()
    for i, k in enumerate(keys.tolist()):
        if k not in seen:
            seen.add(k)
            first_indices.append(i)
    first_indices = torch.tensor(first_indices, dtype=torch.long)
    und_unique = und[:, first_indices]

    # sort lexicographically
    order = torch.argsort(und_unique[0] * (num_nodes + 1) + und_unique[1])
    und_sorted = und_unique[:, order]
    return und_sorted

def compute_forman_ricci_curvature(graph_data, gamma_max_default=1.0):
    """
    Compute balanced Forman-Ricci curvature for each UNDIRECTED edge in a tensorized way.

    Returns:
        Tensor of shape [num_undirected_edges, 3]: [u, v, curvature]
    """
    DENSE_ADJ_THRESHOLD = 300000
    edge_index = graph_data.edge_index
    if edge_index is None or edge_index.numel() == 0:
        return torch.empty((0, 3), dtype=torch.float)

    num_nodes = int(graph_data.num_nodes)
    und_edge_index = _unique_undirected_edge_list(edge_index, num_nodes)  # [2, E]
    E = und_edge_index.shape[1]
    if E == 0:
        return torch.empty((0, 3), dtype=torch.float)

    U = und_edge_index[0]  # [E]
    V = und_edge_index[1]  # [E]

    # Build adjacency - try dense boolean adjacency if not too large (vectorized operations)
    if num_nodes <= DENSE_ADJ_THRESHOLD:
        # Construct dense adjacency (undirected)
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
        # Fill entries from original edge_index (directed or undirected)
        if edge_index.numel() > 0:
            a = edge_index[0].to(torch.long)
            b = edge_index[1].to(torch.long)
            adj[a, b] = True
            adj[b, a] = True
        # Remove self loops just in case
        adj.fill_diagonal_(False)

        # Degrees
        degrees = adj.sum(dim=1).to(torch.int32)  # [N]

        # Triangle counts per undirected edge:
        # triangles for edge (u,v): number of common neighbors = (adj[u] & adj[v]).sum()
        adj_u = adj[U]              # [E, N]
        adj_v = adj[V]              # [E, N]
        common = adj_u & adj_v      # [E, N]
        triangle_counts = common.sum(dim=1).to(torch.int32)  # [E]

        # 4-cycles: we search for k in S_u and w in S_v with A[k,w]==1 and without diagonals u-w or v-k.
        # S_u = neighbors(u) \ neighbors(v) \ {v}
        # S_v = neighbors(v) \ neighbors(u) \ {u}
        S_u_mask = adj_u & (~adj_v)
        S_v_mask = adj_v & (~adj_u)
        # remove endpoints (explicit)
        S_u_mask[torch.arange(E), V] = False
        S_v_mask[torch.arange(E), U] = False

        # Build possible connections between nodes k (rows) and w (cols): need A[k,w]==1
        # A is adj [N,N] ; expand to [1,N,N] then broadcast to [E,N,N]
        A_expand = adj.unsqueeze(0)  # [1, N, N]
        K_mask = S_u_mask.unsqueeze(2)  # [E, N, 1]
        W_mask = S_v_mask.unsqueeze(1)  # [E, 1, N]
        possible = K_mask & A_expand & W_mask  # [E, N, N] where possible[e,k,w]=True if k in S_u, A[k,w], w in S_v

        # Exclude diagonals u-w and v-k: build masks
        # A_u_w is adj[u, w] for each edge e and each w => [E, N]
        A_u_w = adj[U]  # already adj_u: [E, N]
        A_v_k = adj[V]  # already adj_v: [E, N]
        mask_no_diag = (~A_u_w).unsqueeze(1) & (~A_v_k).unsqueeze(2)  # [E, 1, N] & [E, N,1] -> [E, N, N]
        possible = possible & mask_no_diag

        # canonical ordering k < w to avoid double counting
        idx = torch.arange(num_nodes)
        k_idx = idx.view(-1, 1)  # [N,1]
        w_idx = idx.view(1, -1)  # [1,N]
        order_mask = (k_idx < w_idx)  # [N, N]
        possible = possible & order_mask.unsqueeze(0)

        # Now count unique 4-cycles for each edge
        four_counts_matrix = possible  # [E,N,N] boolean
        four_counts = four_counts_matrix.sum(dim=(1, 2)).to(torch.int32)  # [E]
        # For the code you had: you counted side_i and side_j similarly (I used same counts both sides)
        square_counts_i = four_counts.clone()
        square_counts_j = four_counts.clone()

        # Per-node counts of how many 4-cycles pass through a given node (k or w)
        counts_k = four_counts_matrix.sum(dim=2)  # [E, N] counts for k
        counts_w = four_counts_matrix.sum(dim=1)  # [E, N] counts for w
        per_node_counts = counts_k + counts_w     # [E, N] total cycles touching node
        # gamma_max per edge: maximum over nodes; if zero, fallback to gamma_max_default
        gamma_max_vals = per_node_counts.max(dim=1).values.to(torch.float)
        zero_mask = gamma_max_vals == 0.0
        if zero_mask.any():
            gamma_max_vals[zero_mask] = float(gamma_max_default)

        # Now compute ricci curvature vectorized
        du = degrees[U].to(torch.float)  # [E]
        dv = degrees[V].to(torch.float)  # [E]
        tri = triangle_counts.to(torch.float)
        sq_i = square_counts_i.to(torch.float)
        sq_j = square_counts_j.to(torch.float)
        gamma = gamma_max_vals

        # Safe computations: where min(du,dv) <= 1 => ricci = 0
        min_deg = torch.min(du, dv)
        max_deg = torch.max(du, dv)
        term1 = 2.0 / du + 2.0 / dv - 2.0
        term2 = 2.0 * tri / torch.where(max_deg != 0, max_deg, torch.ones_like(max_deg))
        term3 = tri / torch.where(min_deg != 0, min_deg, torch.ones_like(min_deg))
        term4 = ((sq_i + sq_j) / torch.where(max_deg != 0, max_deg, torch.ones_like(max_deg))) / torch.where(gamma > 0, gamma, torch.ones_like(gamma))
        ricci_arr = term1 + term2 + term3 + term4
        ricci_arr = torch.where(min_deg <= 1.0, torch.zeros_like(ricci_arr), ricci_arr)

        # Build output
        out = torch.zeros((E, 3), dtype=torch.float)
        out[:, 0] = U.to(torch.float)
        out[:, 1] = V.to(torch.float)
        out[:, 2] = ricci_arr

        return out

    else:
        print("fallback too large graphs")
        # Fallback: large graph - use neighbor list approach but still keep as vectorized as possible
        # Build neighbor sets (python lists of sets) - less memory but more loops
        neighbors = [set() for _ in range(num_nodes)]
        a = edge_index[0].tolist()
        b = edge_index[1].tolist()
        for i in range(len(a)):
            u = int(a[i]); v = int(b[i])
            if u == v: continue
            neighbors[u].add(v)
            neighbors[v].add(u)
        degrees = torch.tensor([len(neighbors[i]) for i in range(num_nodes)], dtype=torch.int32)

        triangle_counts = torch.zeros(E, dtype=torch.int32)
        square_counts_i = torch.zeros(E, dtype=torch.int32)
        square_counts_j = torch.zeros(E, dtype=torch.int32)
        gamma_max_list = torch.zeros(E, dtype=torch.float)

        for idx in range(E):
            u = int(U[idx].item()); v = int(V[idx].item())
            # triangles
            tri_nodes = neighbors[u].intersection(neighbors[v])
            triangle_counts[idx] = len(tri_nodes)

            # 4-cycles (u - k - w - v)
            S_u = neighbors[u] - neighbors[v] - {v}
            S_v = neighbors[v] - neighbors[u] - {u}
            sq_nodes_counts = {}
            count = 0
            for k in S_u:
                # iterate neighbors of k and check if they are in S_v
                for w in neighbors[k]:
                    if w not in S_v:
                        continue
                    if (w in neighbors[u]) or (k in neighbors[v]):
                        continue
                    if k < w:
                        count += 1
                        sq_nodes_counts[k] = sq_nodes_counts.get(k, 0) + 1
                        sq_nodes_counts[w] = sq_nodes_counts.get(w, 0) + 1
            square_counts_i[idx] = count
            square_counts_j[idx] = count
            if sq_nodes_counts:
                gamma_max_list[idx] = float(max(sq_nodes_counts.values()))
            else:
                gamma_max_list[idx] = float(gamma_max_default)

        # compute curvature vectorized from tensors
        Uf = U.to(torch.float)
        Vf = V.to(torch.float)
        du = degrees[U].to(torch.float)
        dv = degrees[V].to(torch.float)
        tri = triangle_counts.to(torch.float)
        sq_i = square_counts_i.to(torch.float)
        sq_j = square_counts_j.to(torch.float)
        gamma = gamma_max_list

        min_deg = torch.min(du, dv)
        max_deg = torch.max(du, dv)
        term1 = 2.0 / du + 2.0 / dv - 2.0
        term2 = 2.0 * tri / torch.where(max_deg != 0, max_deg, torch.ones_like(max_deg))
        term3 = tri / torch.where(min_deg != 0, min_deg, torch.ones_like(min_deg))
        term4 = ((sq_i + sq_j) / torch.where(max_deg != 0, max_deg, torch.ones_like(max_deg))) / torch.where(gamma > 0, gamma, torch.ones_like(gamma))
        ricci_arr = term1 + term2 + term3 + term4
        ricci_arr = torch.where(min_deg <= 1.0, torch.zeros_like(ricci_arr), ricci_arr)

        out = torch.zeros((E, 3), dtype=torch.float)
        out[:, 0] = Uf
        out[:, 1] = Vf
        out[:, 2] = ricci_arr
        return out


def make_bidirected_edge_index(edge_index):
    """
    Given a directed edge_index [2, num_edges], return a bidirected version:
    for each edge i->j, also add j->i.
    """
    reversed_edges = torch.stack([edge_index[1], edge_index[0]], dim=0)
    bidir_edge_index = torch.cat([edge_index, reversed_edges], dim=1)
    return bidir_edge_index




def line_graph_pyg(graph_data):
    """
    Compute the line graph of a PyG graph.

    Args:
        graph_data (torch_geometric.data.Data): original graph
    
    Returns:
        line_graph_data (torch_geometric.data.Data): line graph with
            num_nodes = number of edges in original graph
            edge_index = edges between edges that share a node
    """
    edge_index = graph_data.edge_index
    num_edges = edge_index.shape[1]
    
    # Each edge becomes a node in the line graph
    nodes_in_linegraph = list(range(num_edges))
    
    # Build adjacency for line graph
    line_edges = []
    for i in range(num_edges):
        u1, v1 = edge_index[:, i]
        for j in range(i+1, num_edges):
            u2, v2 = edge_index[:, j]
            # Connect if edges share a node
            if len({int(u1), int(v1), int(u2), int(v2)}) < 4:
                line_edges.append([i, j])
                line_edges.append([j, i])  # undirected
    
    if line_edges:
        line_edge_index = torch.tensor(line_edges, dtype=torch.long).t()
    else:
        line_edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # Optionally, node features: just zeros
    x_line = torch.zeros((num_edges, graph_data.x.shape[1] if graph_data.x is not None else 1))
    # lg_edge_index_undirected = make_undirected_edge_index(line_edge_index)
    line_graph_data = Data(x=x_line, edge_index=line_edge_index)
    return line_graph_data



from matplotlib.ticker import PercentFormatter



def compute_forman_ricci_for_dataset(dataset):
    """
    Compute the Forman-Ricci curvature for all graphs in a dataset,
    and plot histograms with mean/std.
    """
    all_ricci_curvatures = []
    ricci_all = []
    lg_ricci_all = []
    print("in curv for dataset")
    graph_densities = []
    linegraph_densities = []
    
    ricci_mean = 0 ; ricci_low = 0
    for i, graph_data in enumerate(tqdm(dataset)):
        # if i == 20000:
        #     break

        ricci_curvatures = compute_forman_ricci_curvature(graph_data)
        lg_ricci = compute_forman_ricci_curvature(line_graph_pyg(graph_data))

        graph_curv = ricci_curvatures[:, 2]
        lg_curv = lg_ricci[:, 2]

        min_a = ricci_curvatures.min()
        min_b = lg_ricci.min()

        if min_a < min_b:
            ricci_low += 1
        else:
            ricci_low -= 1
        
        if ricci_curvatures.mean() < lg_ricci.mean():
            ricci_mean += 1
        else:
            ricci_mean -= 1

        ricci_all.append(graph_curv)
        lg_ricci_all.append(lg_curv)

        # Compute edge densities
        N = graph_data.num_nodes
        E = graph_data.edge_index.shape[1] // 2  # undirected
        density = 2 * E / (N * (N - 1)) if N > 1 else 0.0
        graph_densities.append(density)

        lg_N = lg_ricci.shape[0]  # number of edges becomes nodes in line graph
        lg_E = line_graph_pyg(graph_data).edge_index.shape[1] // 2  # undirected
        lg_density = 2 * lg_E / (lg_N * (lg_N - 1)) if lg_N > 1 else 0.0
        linegraph_densities.append(lg_density)

    # Flatten
    ricci_all_flat = torch.cat(ricci_all).numpy()
    lg_ricci_all_flat = torch.cat(lg_ricci_all).numpy()

    # Compute statistics
    r_mean, r_std = np.mean(ricci_all_flat), np.std(ricci_all_flat)
    lg_mean, lg_std = np.mean(lg_ricci_all_flat), np.std(lg_ricci_all_flat)

    print(f"Original graph curvature: mean={r_mean:.4f}, std={r_std:.4f}")
    print(f"Line graph curvature: mean={lg_mean:.4f}, std={lg_std:.4f}")

    # Edge density statistics
    graph_density_mean, graph_density_std = np.mean(graph_densities), np.std(graph_densities)
    lg_density_mean, lg_density_std = np.mean(linegraph_densities), np.std(linegraph_densities)

    print(f"Original graph edge density: mean={graph_density_mean:.4f}, std={graph_density_std:.4f}")
    print(f"Line graph edge density: mean={lg_density_mean:.4f}, std={lg_density_std:.4f}")

    # Histogram (absolute frequency)
    plt.figure(figsize=(10,6))
    sns.histplot(ricci_all_flat, bins=50, color='blue', alpha=0.5, label='Original Graph')
    sns.histplot(lg_ricci_all_flat, bins=50, color='orange', alpha=0.5, label='Line Graph')

    # Overlay mean + std lines
    plt.axvline(r_mean, color='blue', linestyle='--', label=f'Original mean: {r_mean:.2f}')
    plt.axvspan(r_mean - r_std, r_mean + r_std, color='blue', alpha=0.1)

    plt.axvline(lg_mean, color='orange', linestyle='--', label=f'Line Graph mean: {lg_mean:.2f}')
    plt.axvspan(lg_mean - lg_std, lg_mean + lg_std, color='orange', alpha=0.1)

    plt.xlabel("Forman-Ricci Curvature")
    plt.ylabel("Frequency")
    plt.title("Distribution of Forman-Ricci Curvature")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2Histogram (proportion / probability)
    plt.figure(figsize=(10,6))
    sns.histplot(ricci_all_flat, bins=50, color='blue', alpha=0.5, label='Original Graph', stat='probability')
    sns.histplot(lg_ricci_all_flat, bins=50, color='orange', alpha=0.5, label='Line Graph', stat='probability')

    # Overlay mean + std lines again
    plt.axvline(r_mean, color='blue', linestyle='--', label=f'Original mean: {r_mean:.2f}')
    plt.axvspan(r_mean - r_std, r_mean + r_std, color='blue', alpha=0.1)

    plt.axvline(lg_mean, color='orange', linestyle='--', label=f'Line Graph mean: {lg_mean:.2f}')
    plt.axvspan(lg_mean - lg_std, lg_mean + lg_std, color='orange', alpha=0.1)

    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.xlabel("Forman-Ricci Curvature")
    plt.ylabel("Proportion (%)")
    plt.title("Proportion Distribution of Forman-Ricci Curvature")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("ricci mean comparison counter:", ricci_mean)
    print("ricci low comparison counter:", ricci_low)
    return all_ricci_curvatures


import matplotlib.pyplot as plt
import xgi
def draw_pyg_as_xgi(pyg_data_object, ax=None, layout_method='spring', **kwargs):
    """
    Draws a PyTorch Geometric graph as an XGI hypergraph with a consistent layout.

    Args:
        pyg_data_object (torch_geometric.data.Data): The PyG Data object.
        ax (matplotlib.axes.Axes, optional): The axes to draw on. 
                                            If None, a new figure and axes are created.
        layout_method (str): The layout algorithm to use for the plot.
                             e.g., 'spring', 'circular', 'spectral', 'barycenter'.
        **kwargs: Additional keyword arguments to pass to xgi.drawing.draw().
    """
    # 1. Create a graph with only the base edges (dyads) for layout calculation
    edges = set()
    if hasattr(pyg_data_object, 'edge_index'):
        edge_index = pyg_data_object.edge_index
        for i in range(edge_index.size(1)):
            u, v = edge_index[:, i].tolist()
            if u < v:
                edges.add((u, v))
            else:
                edges.add((v, u))
    
    # Create the initial hypergraph with only the dyads
    H = xgi.Hypergraph([list(edge) for edge in edges])

    # 2. Compute the layout on this base hypergraph
    layout_func = getattr(xgi.layout, f"{layout_method}_layout", None)
    if layout_func is None:
        #print(f"Warning: Layout method '{layout_method}' not found. Using 'spring_layout' as fallback.")
        layout_func = xgi.layout.barycenter_spring_layout
    
    pos = layout_func(H,seed=42)
    
    # 3. Add the hyperedges to the hypergraph object
    if hasattr(pyg_data_object, 'hyperedges') and pyg_data_object.hyperedges:
        valid_hyperedges = [he for he in pyg_data_object.hyperedges if len(he) > 0]
        if valid_hyperedges:
            H.add_edges_from(valid_hyperedges)
        # H.add_edges_from(pyg_data_object.hyperedges)
    
    if ax is None:
        fig, ax = plt.subplots()

    # 4. Draw the final hypergraph with hyperedges, using the pre-calculated positions
    xgi.drawing.draw(H, ax=ax, pos=pos, **kwargs)

    ax.set_title(f"XGI graph from PyG Data Object ({layout_method} layout)")
    plt.show()





import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
import networkx as nx
from matplotlib import cm

def draw_pyg_with_edge_colors(pyg_data_object, edge_values=None, ax=None,
                              layout_method='spring', cmap='RdBu_r', edge_width=3, **kwargs):
    # Convert PyG Data to NetworkX graph
    G = nx.DiGraph()  # use nx.Graph() for undirected
    edge_index = pyg_data_object.edge_index
    edges = [(int(u), int(v)) for u, v in edge_index.t().tolist()]
    G.add_edges_from(edges)

    # Layout
    if layout_method == 'spring':
        pos = nx.spring_layout(G, seed=42)
    elif layout_method == 'circular':
        pos = nx.circular_layout(G)
    elif layout_method == 'spectral':
        pos = nx.spectral_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    # Edge colors
    if edge_values is not None:
        edge_values = torch.tensor(edge_values, dtype=torch.float)
        vmin, vmax = edge_values.min().item(), edge_values.max().item()
        norm = (edge_values - vmin) / (vmax - vmin + 1e-8)
        edge_colors = [cm.get_cmap(cmap)(float(val)) for val in norm]
    else:
        edge_colors = 'black'

    # Draw
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))

    nx.draw(G, pos, ax=ax, edge_color=edge_colors, width=edge_width,
            with_labels=True, node_color='lightblue', **kwargs)

    ax.set_title("Graph with edge values colored")
    plt.show()
