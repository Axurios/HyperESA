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
 
# -----------------------------------------
# YOU NEED TO PARALLELIZE THIS  (sorry)
# -----------------------------------------
def compute_forman_ricci_curvature(graph_data, gamma_max_default=1.0):
    """
    Compute balanced Forman-Ricci curvature for each edge using explicit neighbor sets
    and full gamma_max scaling, similar to CUDA version but in pure Python/PyTorch.

    Args:
        graph_data: PyG Data object
        gamma_max_default: fallback for γ_max if no 4-cycles exist

    Returns:
        Tensor of shape [num_edges, 3]: [source, target, curvature]
    """
    edge_index = graph_data.edge_index
    edge_index = make_bidirected_edge_index(edge_index)
    num_nodes = graph_data.num_nodes

    # ---- Compute degrees ----
    degrees = torch.zeros(num_nodes, dtype=torch.int32)
    degrees.scatter_add_(0, edge_index[0], torch.ones(edge_index[0].size(0), dtype=torch.int32))
    degrees.scatter_add_(0, edge_index[1], torch.ones(edge_index[1].size(0), dtype=torch.int32))

    # ---- Build neighbor sets ----
    neighbors = [set() for _ in range(num_nodes)]
    for u, v in edge_index.t().tolist():
        neighbors[u].add(v)
        neighbors[v].add(u)

    # ---- Compute triangles, 4-cycles, and gamma_max ----
    num_edges = edge_index.shape[1]
    triangle_counts = torch.zeros(num_edges, dtype=torch.int32)
    square_counts_i = torch.zeros(num_edges, dtype=torch.int32)
    square_counts_j = torch.zeros(num_edges, dtype=torch.int32)
    gamma_max_list = torch.zeros(num_edges, dtype=torch.float)

    for idx, (u, v) in enumerate(edge_index.t().tolist()):
        # Triangles
        tri = neighbors[u].intersection(neighbors[v])
        triangle_counts[idx] = len(tri)

        # Squares (4-cycles w/o diagonals)
        S1_u_minus_v = neighbors[u] - neighbors[v] - {v}
        S1_v_minus_u = neighbors[v] - neighbors[u] - {u}

        sq_i_nodes = {}
        sq_j_nodes = {}
        count_i = 0
        count_j = 0

        for k in S1_u_minus_v:
            for w in neighbors[k]:
                if w in S1_v_minus_u and (u not in neighbors[w]):
                    count_i += 1
                    sq_i_nodes[k] = sq_i_nodes.get(k, 0) + 1
                    sq_i_nodes[w] = sq_i_nodes.get(w, 0) + 1

        for k in S1_v_minus_u:
            for w in neighbors[k]:
                if w in S1_u_minus_v and (v not in neighbors[w]):
                    count_j += 1
                    sq_j_nodes[k] = sq_j_nodes.get(k, 0) + 1
                    sq_j_nodes[w] = sq_j_nodes.get(w, 0) + 1

        square_counts_i[idx] = count_i
        square_counts_j[idx] = count_j

        # gamma_max = max number of 4-cycles passing through a common node
        all_counts = list(sq_i_nodes.values()) + list(sq_j_nodes.values())
        gamma_max = max(all_counts) if all_counts else gamma_max_default
        gamma_max_list[idx] = gamma_max

    # ---- Compute balanced Forman-Ricci curvature ----
    ricci_curvature = torch.zeros(num_edges, dtype=torch.float)

    for idx, (u, v) in enumerate(edge_index.t().tolist()):
        du = degrees[u].item()
        dv = degrees[v].item()
        tri = triangle_counts[idx].item()
        sq_i = square_counts_i[idx].item()
        sq_j = square_counts_j[idx].item()
        gamma = gamma_max_list[idx].item()


        if min(du, dv) <= 1:
            ricci_curvature[idx] = 0
            continue

        # Degree term
        term1 = 2 / du + 2 / dv - 2

        # Triangle term
        term2 = 2 * tri / (max(du, dv)) if max(du, dv) != 0 else 0
        term3 = tri / (min(du, dv)) if min(du, dv) != 0 else 0

        # 4-cycle term with gamma_max scaling
        term4 = ((sq_i + sq_j) / max(du, dv)) / gamma if gamma > 0 else 0

        ricci_curvature[idx] = term1 + term2 + term3 + term4

    # ---- Build final tensor: [source, target, Ricci] ----
    edges_with_curvature = torch.zeros((num_edges, 3), dtype=torch.float)
    edges_with_curvature[:, 0:2] = edge_index.t().float()
    edges_with_curvature[:, 2] = ricci_curvature
    # edges_with_curvature[:, 3] = triangle_counts

    return edges_with_curvature




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









def compute_forman_ricci_for_dataset(dataset):
    """
    Compute the Forman-Ricci curvature for all graphs in a dataset.
    
    Args:
        dataset: A list of PyG Data objects representing graphs.
        
    Returns:
        ricci_curvatures: A list of lists, where each inner list contains the Forman-Ricci curvature 
                           of the edges in the corresponding graph.
    """
    all_ricci_curvatures = []
    print("in curv for dataset")
    
    ricci_mean = 0 ; ricci_low = 0
    for graph_data in tqdm(dataset):
        ricci_curvatures = compute_forman_ricci_curvature(graph_data)

        lg_ricci = compute_forman_ricci_curvature(line_graph_pyg(graph_data))
        graph_curv = ricci_curvatures[:, 2] ; lg_curv = lg_ricci[:, 2]
        # draw_pyg_with_edge_colors(graph_data, edge_values=graph_curv, layout_method='spring')
        # print(ricci_curvatures) ; print(lg_ricci)
        # all_ricci_curvatures.append(ricci_curvatures)
        min_a = ricci_curvatures.min()
        min_b = lg_ricci.min()

        if min_a < min_b:
            # print("ricci_curvatures has the smaller value:", min_a.item())
            ricci_low += 1
        else:
            # print("lg_ricci has the smaller value:", min_b.item())
            ricci_low -= 1
        
        if ricci_curvatures.mean() < lg_ricci.mean():
            # print("ricci_curvatures has the smaller mean")
            ricci_mean += 1
        else:
            # print("lg_ricci has the smaller sum")
            ricci_mean -= 1
    
    print("ricci mean", ricci_mean) ; print("ricci_low", ricci_low)
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
