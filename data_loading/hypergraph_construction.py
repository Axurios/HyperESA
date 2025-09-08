import numpy as np
#import pandas as pd
#import os
import torch
import collections
import matplotlib.pyplot as plt
import random as rd
# import torch_geometric
# from torch_geometric.utils import degree, dense_to_sparse
# from torch_geometric.data import Data, InMemoryDataset
# from torch.utils.data import random_split
# from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import to_networkx, k_hop_subgraph
from tqdm.auto import tqdm
import networkx as nx
# from gspan_mining.gspan import gSpan
import xgi
from torch_geometric.data import InMemoryDataset
#import torch.utils.data as data_utils
import random



# def add_hyper_edges_to_dataset_no_vocab(dataset, min_cycle_len=3, max_cycle_len=9):
#     """
#     Adds hyperedges to graphs by directly detecting simple cycles.

#     Args:
#         dataset (list): The list of PyG Data objects.
#         min_cycle_len (int): The minimum number of nodes in a cycle (e.g., 3 for a triangle).
#         max_cycle_len (int): The maximum number of nodes in a cycle.

#     Returns:
#         list: A new dataset with added hyperedges.
#     """
#     new_dataset = dataset.clone()
    
#     for new_data in tqdm(new_dataset, desc="Adding hyper-edges from cycles"):
        
#         # Convert the PyG graph to an undirected NetworkX graph
#         nx_graph = to_networkx(new_data, to_undirected=True)
        
#         found_hyperedges_set = set()

#         # Find all simple cycles in the graph
#         # This is more direct than the k-hop subgraph approach
#         for cycle in nx.simple_cycles(nx_graph):
#             # Check if the cycle length is within the desired range
#             cycle_len = len(cycle)
#             if min_cycle_len <= cycle_len <= max_cycle_len:
#                 # Normalise the hyperedge by sorting the nodes
#                 new_hyperedge = sorted(cycle)
#                 found_hyperedges_set.add(frozenset(new_hyperedge))
        
#         # Assign the unique hyperedges to the new graph
#         new_data.hyperedges = [list(h) for h in found_hyperedges_set]
        
#     return new_dataset



# def add_hyper_edges_to_dataset_no_vocab(dataset, min_cycle_len=3, max_cycle_len=9):
#     """
#     Adds hyperedges to graphs by directly detecting simple cycles.

#     Args:
#         dataset (torch_geometric.data.Dataset): The PyG Dataset object.
#         min_cycle_len (int): The minimum number of nodes in a cycle (e.g., 3 for a triangle).
#         max_cycle_len (int): The maximum number of nodes in a cycle.

#     Returns:
#         torch_geometric.data.Dataset: A new Dataset object with added hyperedges.
#     """
#     # Loop over the individual Data objects in the dataset
#     total_hyperedges = 0
#     for data in tqdm(dataset, desc="Adding hyper-edges from cycles"):
        
#         # CORRECT: Clone each individual Data object
#         #new_data = data.clone()
        
#         # Convert the PyG graph to an undirected NetworkX graph
#         nx_graph = to_networkx(data, to_undirected=True)
        
#         found_hyperedges_set = set()

#         for cycle in nx.simple_cycles(nx_graph):
#             cycle_len = len(cycle)
#             if min_cycle_len <= cycle_len <= max_cycle_len:
#                 new_hyperedge = sorted(cycle)
#                 found_hyperedges_set.add(frozenset(new_hyperedge))
#         hyperedges_list = [list(h) for h in found_hyperedges_set]
#         data.hyperedges = hyperedges_list
#         total_hyperedges += len(hyperedges_list)

#     num_graphs = len(dataset)
#     if num_graphs > 0:
#         avg_hyperedges = total_hyperedges / num_graphs
#         print(f"\nAverage number of hyperedges per graph: {avg_hyperedges:.2f}")
#     else:
#         print("\nNo graphs found in the dataset.")
    
#     return dataset

# def add_hyper_edges_to_dataset_no_vocab(dataset, min_cycle_len=3, max_cycle_len=9):
#     """
#     Adds hyperedges to graphs by directly detecting simple cycles.
#     The hyperedges are represented by the node features (x) of the nodes
#     in the cycle, rather than their local indices.

#     Args:
#         dataset (torch_geometric.data.Dataset): The PyG Dataset object.
#         min_cycle_len (int): The minimum number of nodes in a cycle (e.g., 3 for a triangle).
#         max_cycle_len (int): The maximum number of nodes in a cycle.

#     Returns:
#         torch_geometric.data.Dataset: A new Dataset object with added hyperedges,
#                                       where hyperedges are lists of node feature tensors.
#     """
#     total_hyperedges = 0
#     for data in tqdm(dataset, desc="Adding hyper-edges from cycles"):
        
#         # Convert the PyG graph to an undirected NetworkX graph
#         nx_graph = to_networkx(data, to_undirected=True)
        
#         found_hyperedges_set = set()

#         for cycle in nx.simple_cycles(nx_graph):
#             cycle_len = len(cycle)
#             if min_cycle_len <= cycle_len <= max_cycle_len:
#                 # Get the features (x) for each node in the cycle
#                 hyperedge_features = [data.x[node_idx] for node_idx in cycle]
                
#                 # To handle duplicate hyperedges, we need a hashable representation.
#                 # We can convert the list of tensors to a tuple of tuples.
#                 # Sorting by some criteria (e.g., the first element of the tensor)
#                 # ensures that the hyperedge is unique regardless of the cycle's starting node.
#                 sorted_hyperedge_features = tuple(
#                     tuple(t.tolist()) for t in sorted(hyperedge_features, key=lambda t: t.tolist())
#                 )
                
#                 found_hyperedges_set.add(sorted_hyperedge_features)

#         # Convert the frozenset of tuples back to a list of lists of tensors
#         hyperedges_list = [
#             [torch.tensor(item) for item in hyperedge_tuple]
#             for hyperedge_tuple in found_hyperedges_set
#         ]
        
#         data.hyperedges = hyperedges_list
#         total_hyperedges += len(hyperedges_list)

#     num_graphs = len(dataset)
#     if num_graphs > 0:
#         avg_hyperedges = total_hyperedges / num_graphs
#         print(f"\nAverage number of hyperedges per graph: {avg_hyperedges:.2f}")
#     else:
#         print("\nNo graphs found in the dataset.")
    
#     return dataset


def add_hyper_edges_to_dataset_no_vocab(dataset, min_cycle_len=3, max_cycle_len=9):
    """
    Adds hyperedges to graphs by directly detecting simple cycles.
    It adds two new attributes:
    1. `data.hyperedges`: a list of lists of node indices, representing the hyperedges.
    2. `data.hyperedge_features`: a list of lists of node feature tensors, corresponding
       to the nodes in each hyperedge.

    Args:
        dataset (torch_geometric.data.Dataset): The PyG Dataset object.
        min_cycle_len (int): The minimum number of nodes in a cycle (e.g., 3 for a triangle).
        max_cycle_len (int): The maximum number of nodes in a cycle.

    Returns:
        torch_geometric.data.Dataset: A new Dataset object with added hyperedges
                                      and their corresponding features.
    """
    total_hyperedges = 0
    for data in tqdm(dataset, desc="Adding hyper-edges from cycles"):
        
        # Convert the PyG graph to an undirected NetworkX graph
        nx_graph = to_networkx(data, to_undirected=True)
        
        found_hyperedges_set = set()
        hyperedges_list = []
        #hyperedge_features_list = []

        for cycle in nx.simple_cycles(nx_graph):
            cycle_len = len(cycle)
            if min_cycle_len <= cycle_len <= max_cycle_len:
                # Use a frozenset of the sorted cycle to ensure uniqueness
                # regardless of starting node or direction.
                hyperedge_nodes_set = frozenset(sorted(cycle))
                
                if hyperedge_nodes_set not in found_hyperedges_set:
                    found_hyperedges_set.add(hyperedge_nodes_set)
                    
                    # Store the hyperedge as a list of node indices
                    hyperedge_nodes = sorted(list(hyperedge_nodes_set))
                    hyperedges_list.append(hyperedge_nodes)
                    
                    # # Get the features for each node in the hyperedge
                    # hyperedge_features = [data.x[node_idx] for node_idx in hyperedge_nodes]
                    # hyperedge_features_list.append(hyperedge_features)

        data.hyperedges = hyperedges_list
        #data.hyperedge_features = hyperedge_features_list
        total_hyperedges += len(hyperedges_list)

    num_graphs = len(dataset)
    if num_graphs > 0:
        avg_hyperedges = total_hyperedges / num_graphs
        print(f"\nAverage number of hyperedges per graph: {avg_hyperedges:.2f}")
    else:
        print("\nNo graphs found in the dataset.")
    
    return dataset



def add_one_random_subgroup_hyperedge(dataset, min_size=3, max_size=9):
    """
    Adds one random hyperedge to each graph by selecting a random subgroup of nodes.
    The size of the subgroup is within the specified range.
    It adds two new attributes:
    1. `data.hyperedges`: a list of lists of node indices, representing the hyperedges.
    2. `data.hyperedge_features`: a list of lists of node feature tensors, corresponding
        to the nodes in each hyperedge.

    Args:
        dataset (torch_geometric.data.Dataset): The PyG Dataset object.
        min_size (int): The minimum number of nodes in the subgroup.
        max_size (int): The maximum number of nodes in the subgroup.

    Returns:
        torch_geometric.data.Dataset: A new Dataset object with one random hyperedge
                                      and its corresponding features added to each graph.
    """
    total_hyperedges = 0
    for data in tqdm(dataset, desc="Adding one random subgroup hyper-edge"):
        num_nodes = data.num_nodes
        
        # Ensure the graph has enough nodes to form a subgroup of the minimum size
        if num_nodes >= min_size:
            # Determine a random size for the subgroup within the specified range
            subgroup_size = random.randint(min_size, min(max_size, num_nodes))
            
            # Select a random sample of nodes to form the hyperedge
            random_nodes = random.sample(range(num_nodes), subgroup_size)
            random_nodes.sort()  # Sort for consistency

            data.hyperedges = [random_nodes]
            
            # Get the features for each node in the hyperedge
            #hyperedge_features = [data.x[node_idx] for node_idx in random_nodes]
            #data.hyperedge_features = [hyperedge_features]
            
            total_hyperedges += 1
        else:
            # If the graph is too small, add no hyperedges
            data.hyperedges = []
            #data.hyperedge_features = []

    num_graphs = len(dataset)
    if num_graphs > 0:
        avg_hyperedges = total_hyperedges / num_graphs
        print(f"\nAverage number of hyperedges per graph: {avg_hyperedges:.2f}")
    else:
        print("\nNo graphs found in the dataset.")
    
    return dataset

# def draw_graph(data_point):
#     """
#     Draws a graph given a data object from the dataset.
#     This function should work with the raw, unscaled data.
#     """
#     # Extract graph information
#     x = data_point.x  # Node features
#     edge_index = data_point.edge_index
#     y = data_point.y  # The target value (unscaled)
    
#     # Use a library like NetworkX and Matplotlib to draw the graph
#     G = to_networkx(data_point, to_undirected=True)
#     plt.figure(figsize=(8, 6))
#     nx.draw(G, with_labels=True)
#     plt.title(f"Graph with target: {y.item()}")
#     plt.show()
    
#     print(f"Drawing a graph with target value: {y.item()}")
#     print(f"Number of nodes: {data_point.num_nodes}")
#     print(f"Number of edges: {data_point.num_edges}")





# def add_random_hyperedge(data_point, num_nodes_in_hyperedge=3):
#     # (La même fonction que précédemment)
#     all_nodes = list(range(data_point.num_nodes))
#     if len(all_nodes) < num_nodes_in_hyperedge:
#         return data_point
    
#     random_nodes = rd.sample(all_nodes, num_nodes_in_hyperedge)
    
#     if not hasattr(data_point, 'hyperedges') or data_point.hyperedges is None:
#         data_point.hyperedges = []
        
#     data_point.hyperedges.append(random_nodes)
    
#     print(f"Hyperarête ajoutée : {random_nodes}")
#     return data_point







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
        H.add_edges_from(pyg_data_object.hyperedges)
    
    if ax is None:
        fig, ax = plt.subplots()

    # 4. Draw the final hypergraph with hyperedges, using the pre-calculated positions
    xgi.drawing.draw(H, ax=ax, pos=pos, **kwargs)

    ax.set_title(f"XGI graph from PyG Data Object ({layout_method} layout)")
    plt.show()



# def draw_pyg_as_xgi(pyg_data_object, ax=None, layout_method='spring', **kwargs):
#     """
#     Draws a PyTorch Geometric graph as an XGI hypergraph with a consistent layout.
#     This version is modified to handle hyperedges represented by node features (tensors).

#     Args:
#         pyg_data_object (torch_geometric.data.Data): The PyG Data object.
#         ax (matplotlib.axes.Axes, optional): The axes to draw on.
#                                                 If None, a new figure and axes are created.
#         layout_method (str): The layout algorithm to use for the plot.
#                              e.g., 'spring', 'circular', 'spectral', 'barycenter'.
#         **kwargs: Additional keyword arguments to pass to xgi.drawing.draw().
#     """
#     # 1. Create a graph with only the base edges (dyads) for layout calculation
#     edges = set()
#     if hasattr(pyg_data_object, 'edge_index'):
#         edge_index = pyg_data_object.edge_index
#         for i in range(edge_index.size(1)):
#             u, v = edge_index[:, i].tolist()
#             if u < v:
#                 edges.add((u, v))
#             else:
#                 edges.add((v, u))
    
#     # Create the initial hypergraph with only the dyads
#     H = xgi.Hypergraph([list(edge) for edge in edges])

#     # 2. Compute the layout on this base hypergraph
#     layout_func = getattr(xgi.layout, f"{layout_method}_layout", None)
#     if layout_func is None:
#         #print(f"Warning: Layout method '{layout_method}' not found. Using 'spring_layout' as fallback.")
#         layout_func = xgi.layout.barycenter_spring_layout
    
#     pos = layout_func(H, seed=42)
    
#     # 3. Add the hyperedges to the hypergraph object
#     if hasattr(pyg_data_object, 'hyperedges') and pyg_data_object.hyperedges:
#         # We need to map the feature tensors back to their node indices.
#         # This requires creating a mapping from the feature tensor to the node index.
#         # We assume the features are unique for each node.
        
#         # Invert the mapping: from feature tensor (as a tuple) to node index
#         node_features_to_idx = {
#             tuple(pyg_data_object.x[i].tolist()): i
#             for i in range(pyg_data_object.num_nodes)
#         }
        
#         hyperedge_indices = []
#         for hyperedge_features in pyg_data_object.hyperedges:
#             # For each hyperedge (which is a list of feature tensors),
#             # convert it back to a list of node indices.
#             indices = [node_features_to_idx[tuple(feat.tolist())] for feat in hyperedge_features]
#             hyperedge_indices.append(indices)
        
#         # Add the hyperedges to the hypergraph using the node indices
#         H.add_edges_from(hyperedge_indices)
    
#     if ax is None:
#         fig, ax = plt.subplots()

#     # 4. Draw the final hypergraph with hyperedges, using the pre-calculated positions
#     xgi.drawing.draw(H, ax=ax, pos=pos, **kwargs)

#     ax.set_title(f"XGI graph from PyG Data Object ({layout_method} layout)")
#     plt.show()






def separated_dataset_to_model_compatible(dataset_with_hyperedges_field):
    # create from the hyperedges fields, those h-edges features
    # concatenate them to the edges ones

    # compute the adj of those hyperedges (linegraph)
    # put info on the edges adj.  
    pass

















































# def vocab_in_graph(vocab, node_set):
#     if vocab.shape[1] != node_set.shape[1]:
#         print("Error: The number of features (columns) must be the same for both matrices.")
#         return False
    
#     if vocab.size == 0:
#         return True
    
#     vocab_sorted = np.ascontiguousarray(vocab).view([('', vocab.dtype)] * vocab.shape[1])
#     node_set_sorted = np.ascontiguousarray(node_set).view([('', node_set.dtype)] * node_set.shape[1])
    
#     vocab_sorted.sort(order=vocab_sorted.dtype.names)
#     node_set_sorted.sort(order=node_set_sorted.dtype.names)
    
#     is_in_set = np.isin(vocab_sorted, node_set_sorted)
#     return np.all(is_in_set)






# def get_subgraph_vocabulary(
#     dataset, 
#     max_vocab_size=64, 
#     sample_percentage=0.2,
#     num_hops=4
# ):
#     motif_counts = collections.Counter()
#     all_indices = dataset.indices
#     num_to_sample = int(len(all_indices) * sample_percentage)
#     if num_to_sample == 0 and len(all_indices) > 0:
#         num_to_sample = 1
    
#     sampled_indices = rd.sample(all_indices, k=num_to_sample)
    
#     print(f"Sampling {num_to_sample} graphs for vocabulary mining (num_hops={num_hops}).")
    
#     for idx in tqdm(sampled_indices, desc="Mining subgraph motifs"):
#         data = dataset.dataset[idx]
#         if data.x is None:
#             continue
        
#         for node_idx in range(data.num_nodes):
#             # Ajout de l'argument 'num_nodes'
#             subset, _, _, _ = k_hop_subgraph(
#                 node_idx, 
#                 num_hops=num_hops, 
#                 edge_index=data.edge_index, 
#                 relabel_nodes=False,
#                 num_nodes=data.num_nodes  # Correction ici
#             )
            
#             if len(subset) > 2:
#                 subgraph_features = data.x[subset]
#                 sorted_features_tuple = tuple(
#                     torch.sort(subgraph_features, dim=0).values.flatten().tolist()
#                 )
                
#                 motif = frozenset(sorted_features_tuple)
#                 motif_counts[motif] += 1
            
#     most_common_motifs = [motif for motif, count in motif_counts.most_common(max_vocab_size)]
#     print(f"\nSubgraph vocabulary found ({len(most_common_motifs)} motifs).")
#     return most_common_motifs





























# def add_hyper_edges_to_dataset(dataset, hyperedge_vocabulary, num_hops=3):
#     new_dataset = []
#     vocab_set = set(hyperedge_vocabulary)

#     for data in tqdm(dataset, desc="Adding hyper-edges"):
#         new_data = data.clone()
#         if not hasattr(new_data, 'hyperedges') or new_data.hyperedges is None:
#             new_data.hyperedges = []
        
#         for node_idx in range(new_data.num_nodes):
#             # Ajout de l'argument 'num_nodes'
#             subset, _, _, _ = k_hop_subgraph(
#                 node_idx, 
#                 num_hops=num_hops, 
#                 edge_index=new_data.edge_index, 
#                 relabel_nodes=False,
#                 num_nodes=new_data.num_nodes # Correction ici
#             )
            
#             if len(subset) > 2:
#                 subgraph_features = data.x[subset]
#                 sorted_features_tuple = tuple(
#                     torch.sort(subgraph_features, dim=0).values.flatten().tolist()
#                 )
                
#                 motif = frozenset(sorted_features_tuple)

#                 if motif in vocab_set:
#                     new_hyperedge = list(subset.tolist())
#                     if len(new_hyperedge) > 1:
#                         new_data.hyperedges.append(new_hyperedge)
                    
#         new_dataset.append(new_data)
        
#     return new_dataset

# def add_hyper_edges_to_dataset(dataset, hyperedge_vocabulary, num_hops=3):
#     new_dataset = []
#     vocab_set = set(hyperedge_vocabulary)
#     print("doing add_hyper_edges")
#     print(vocab_set)

#     for data in tqdm(dataset, desc="Adding hyper-edges"):
#         new_data = data.clone()
        
#         # Utiliser un ensemble pour stocker les hyperarêtes uniques
#         found_hyperedges_set = set()
        
#         for node_idx in range(new_data.num_nodes):
#             subset, _, _, _ = k_hop_subgraph(
#                 node_idx, 
#                 num_hops=num_hops, 
#                 edge_index=new_data.edge_index, 
#                 relabel_nodes=False,
#                 num_nodes=new_data.num_nodes
#             )
            
#             if len(subset) > 2:
#                 subgraph_features = data.x[subset]
#                 sorted_features_tuple = tuple(
#                     torch.sort(subgraph_features, dim=0).values.flatten().tolist()
#                 )
#                 motif = frozenset(sorted_features_tuple)

#                 if motif in vocab_set:
#                     new_hyperedge = list(subset.tolist())
#                     if len(new_hyperedge) > 1:
#                         # Convertir la liste en un frozenset avant de l'ajouter
#                         # Cela garantit l'unicité
#                         found_hyperedges_set.add(frozenset(new_hyperedge))
        
#         # Convertir l'ensemble de frozensets en une liste de listes
#         new_data.hyperedges = [list(h) for h in found_hyperedges_set]
        
#         new_dataset.append(new_data)
        
#     return new_dataset


# def add_hyper_edges_to_dataset_no_vocab(dataset, num_hops=3):
#     """
#     Ajoute des hyperarêtes aux graphes en se basant sur la détection
#     de sous-structures condensées (non-linéaires).
    
#     Args:
#         dataset (list): La liste d'objets PyG Data.
#         num_hops (int): Le nombre de sauts pour trouver les sous-structures.
        
#     Returns:
#         list: Un nouveau dataset avec les hyperarêtes ajoutées.
#     """
#     new_dataset = []

#     for data in tqdm(dataset, desc="Adding hyper-edges for condensed structures"):
#         new_data = data.clone()
#         found_hyperedges_set = set()

#         for node_idx in range(new_data.num_nodes):
#             # Trouver le sous-graphe k-hop
#             subset, edge_index_subgraph, mapping, edge_mask = k_hop_subgraph(
#                 node_idx, 
#                 num_hops=num_hops, 
#                 edge_index=new_data.edge_index, 
#                 relabel_nodes=False,
#                 num_nodes=new_data.num_nodes
#             )

#             # Vérifier si le sous-graphe est "condensé" (contient plus de 2 nœuds et au moins un cycle)
#             num_nodes_in_subgraph = len(subset)
#             num_edges_in_subgraph = edge_index_subgraph.size(1) // 2  # Divisé par 2 car les arêtes sont bi-directionnelles

#             # Critère pour une structure non-linéaire :
#             # 1. Plus de 2 nœuds pour éviter les arêtes simples
#             # 2. Le nombre d'arêtes est supérieur au nombre de nœuds - 1 (critère pour un cycle dans un graphe connexe)
#             is_condensed = num_nodes_in_subgraph > 2 and num_edges_in_subgraph > (num_nodes_in_subgraph - 1)

#             if is_condensed:
#                 new_hyperedge = list(subset.tolist())
#                 # Ajouter l'hyperarête à l'ensemble pour garantir l'unicité
#                 found_hyperedges_set.add(frozenset(new_hyperedge))
        
#         # Attribuer les hyperarêtes uniques au nouveau graphe
#         new_data.hyperedges = [list(h) for h in found_hyperedges_set]
#         new_dataset.append(new_data)
        
#     return new_dataset



# def add_hyper_edges_to_dataset_no_vocab(dataset, num_hops=4):
#     """
#     Adds hyperedges to graphs by detecting condensed (closed and small)
#     substructures.

#     Args:
#         dataset (list): The list of PyG Data objects.
#         num_hops (int): The number of hops to find the substructures.

#     Returns:
#         list: A new dataset with added hyperedges.
#     """
#     new_dataset = []
    
#     for data in tqdm(dataset, desc="Adding hyper-edges for condensed structures"):
#         new_data = data.clone()
#         found_hyperedges_set = set()
        
#         # Convert the PyG graph to a NetworkX graph for easier cycle detection
#         nx_graph = to_networkx(data, to_undirected=True)

#         for node_idx in range(new_data.num_nodes):
#             # Find the k-hop subgraph
#             subset, edge_index_subgraph, mapping, edge_mask = k_hop_subgraph(
#                 node_idx, 
#                 num_hops=num_hops, 
#                 edge_index=new_data.edge_index, 
#                 relabel_nodes=False,
#                 num_nodes=new_data.num_nodes
#             )

#             # Convert the subgraph into a NetworkX object for analysis
#             subgraph_nx = nx_graph.subgraph(subset.tolist())

#             # Stricter criteria for a "closed" and "small" structure:
#             num_nodes_in_subgraph = len(subset)
            
#             # The new condition is added here.
#             # We check if the number of nodes is less than 6.
#             if 3 <= num_nodes_in_subgraph < 10:
#                 is_condensed = False
#                 # Check for connectivity
#                 if nx.is_connected(subgraph_nx):
#                     # Check for cycles (a connected graph has a cycle if edges >= nodes)
#                     if subgraph_nx.number_of_edges() >= subgraph_nx.number_of_nodes():
#                         is_condensed = True
                        
#                 if is_condensed:
#                     # Normalize the hyperedge by sorting the nodes to ensure uniqueness
#                     new_hyperedge = sorted(subset.tolist())
#                     found_hyperedges_set.add(frozenset(new_hyperedge))
        
#         # Assign the unique hyperedges to the new graph
#         new_data.hyperedges = [list(h) for h in found_hyperedges_set]
#         new_dataset.append(new_data)
        
#     return new_dataset

# def add_hyper_edges_to_dataset_no_vocab(dataset, num_hops=4):
#     """
#     Adds hyperedges to graphs by detecting condensed (closed and small)
#     substructures.

#     Args:
#         dataset (list): The list of PyG Data objects.
#         num_hops (int): The number of hops to find the substructures.

#     Returns:
#         list: A new dataset with added hyperedges.
#     """
#     new_dataset = []
    
#     for data in tqdm(dataset, desc="Adding hyper-edges for condensed structures"):
#         new_data = data.clone()
#         found_hyperedges_set = set()
        
#         # Convert the PyG graph to a NetworkX graph for easier cycle detection
#         nx_graph = to_networkx(data, to_undirected=True)

#         for node_idx in range(new_data.num_nodes):
#             # Find the k-hop subgraph
#             subset, edge_index_subgraph, mapping, edge_mask = k_hop_subgraph(
#                 node_idx, 
#                 num_hops=num_hops, 
#                 edge_index=new_data.edge_index, 
#                 relabel_nodes=False,
#                 num_nodes=new_data.num_nodes
#             )

#             # Convert the subgraph into a NetworkX object for analysis
#             subgraph_nx = nx_graph.subgraph(subset.tolist())

#             # Refined criteria for a "sensible" structure:
#             num_nodes_in_subgraph = len(subset)
#             num_edges_in_subgraph = subgraph_nx.number_of_edges()

#             is_condensed = False
            
#             # Condition 1: Number of nodes is small (3 to 6)
#             if 3 <= num_nodes_in_subgraph < 6:
#                 # Condition 2: Check for connectivity and cycles
#                 if nx.is_connected(subgraph_nx):
#                     # Condition 3 (NEW): Check for high density.
#                     # Density = #edges / (#nodes * (#nodes - 1) / 2)
#                     max_possible_edges = num_nodes_in_subgraph * (num_nodes_in_subgraph - 1) / 2
#                     density = num_edges_in_subgraph / max_possible_edges

#                     # A good heuristic for "strongly connected" is a density above a certain threshold.
#                     # For a simple cycle, density is low (~2/n). For a clique, it's 1.0.
#                     # A threshold of 0.5 is a good starting point for "strongly connected".
#                     if density > 0.5:
#                         is_condensed = True
            
#             # As a simpler alternative, you could also check directly for cycles,
#             # but the density check is more robust for finding "strongly connected groups"
#             # if not is_condensed and nx.is_connected(subgraph_nx) and subgraph_nx.number_of_edges() >= subgraph_nx.number_of_nodes():
#             #    is_condensed = True
                        
#             if is_condensed:
#                 # Normalize the hyperedge by sorting the nodes to ensure uniqueness
#                 new_hyperedge = sorted(subset.tolist())
#                 found_hyperedges_set.add(frozenset(new_hyperedge))
        
#         # Assign the unique hyperedges to the new graph
#         new_data.hyperedges = [list(h) for h in found_hyperedges_set]
#         new_dataset.append(new_data)
        
#     return new_dataset


# def draw_pyg_as_xgi(pyg_data_object, ax=None, layout_method='spring', **kwargs):
#     """
#     Draws a PyTorch Geometric graph as an XGI hypergraph with a consistent layout.

#     Args:
#         pyg_data_object (torch_geometric.data.Data): The PyG Data object.
#         ax (matplotlib.axes.Axes, optional): The axes to draw on. 
#                                             If None, a new figure and axes are created.
#         layout_method (str): The layout algorithm to use for the plot.
#                              e.g., 'spring', 'circular', 'spectral'.
#         **kwargs: Additional keyword arguments to pass to xgi.drawing.draw().
#     """
#     # 1. Create a graph with only the base edges (dyads) for layout calculation
#     edges = set()
#     if hasattr(pyg_data_object, 'edge_index'):
#         edge_index = pyg_data_object.edge_index
#         for i in range(edge_index.size(1)):
#             u, v = edge_index[:, i].tolist()
#             if u < v:
#                 edges.add((u, v))
#             else:
#                 edges.add((v, u))
    
#     # Create the initial hypergraph with only the dyads
#     H = xgi.Hypergraph([list(edge) for edge in edges])

#     # 2. Compute the layout on this base hypergraph
#     # This is the crucial step for consistency.
#     layout_func = getattr(xgi.layout, f"{layout_method}_layout", None)
#     if layout_func is None:
#         # print(f"Warning: Layout method '{layout_method}' not found. Using 'spring_layout' as fallback.")
#         layout_func = xgi.layout.circular_layout
    
#     pos = layout_func(H)
    
#     # 3. Add the hyperedges to the hypergraph object
#     if hasattr(pyg_data_object, 'hyperedges') and pyg_data_object.hyperedges:
#         H.add_edges_from(pyg_data_object.hyperedges)
    
#     if ax is None:
#         fig, ax = plt.subplots()

#     # 4. Draw the final hypergraph with hyperedges, using the pre-calculated positions
#     xgi.drawing.draw(H, ax=ax, pos=pos, **kwargs)

#     ax.set_title(f"XGI graph from PyG Data Object ({layout_method} layout)")
#     plt.show()


# def draw_pyg_as_xgi_circular(pyg_data_object, ax=None, layout_method='spring', **kwargs):
#     """
#     Draws a PyTorch Geometric graph as an XGI hypergraph.

#     Args:
#         pyg_data_object (torch_geometric.data.Data): The PyG Data object.
#         ax (matplotlib.axes.Axes, optional): The axes to draw on. 
#                                               If None, a new figure and axes are created.
#         **kwargs: Additional keyword arguments to pass to xgi.draw().
#     """
#     # Extract the edge index from the PyG data object
#     edge_index = pyg_data_object.edge_index

#     # Convert the bi-directional edge_index to a unique list of edges
#     edges = set()
#     for i in range(edge_index.size(1)):
#         u, v = edge_index[:, i].tolist()
#         # Ensure consistent order to avoid duplicates (e.g., (0, 1) and (1, 0))
#         if u < v:
#             edges.add((u, v))
#         else:
#             edges.add((v, u))
    
#     # Format the edges into a list of lists for XGI
#     edge_list = [list(edge) for edge in edges]

#     if hasattr(pyg_data_object, 'hyperedges') and pyg_data_object.hyperedges:
#         edge_list.extend(pyg_data_object.hyperedges)
    
#     # 3. Créer l'objet XGI Hypergraph avec l'ensemble des arêtes et hyperarêtes

#     # Create an XGI Hypergraph object
#     H = xgi.Hypergraph(edge_list)
    
#     if ax is None:
#         _, ax = plt.subplots()
    
#     layout_func = getattr(xgi.layout, f"{layout_method}_layout", None)
    
#     if layout_func is None:
#         # Fallback if the requested layout is not found
#         #print(f"Warning: Layout method '{layout_method}' not found. Using 'spring_layout' as fallback.")
#         layout_func = xgi.layout.circular_layout
    
#     pos = layout_func(H)
#     # Draw the hypergraph using XGI
#     xgi.draw(H, ax=ax, pos=pos, **kwargs)

#     # You can add a title or other plot customizations here
#     ax.set_title(f"XGI graph from PyG Data Object")
#     plt.show()



# def get_subgraph_vocabulary(dataset: data_utils.Subset, max_subgraph_size=3, max_vocab_size=30, sample_percentage=0.2):
#     """
#     Détermine les motifs de nœuds fréquents à partir d'un échantillon
#     du dataset pour construire un vocabulaire.
    
#     Args:
#         dataset (data_utils.Subset): Le subset d'objets PyTorch Geometric Data.
#         max_subgraph_size (int): La taille maximale du motif.
#         max_vocab_size (int): La taille maximale du vocabulaire de motifs.
#         sample_percentage (float): La proportion du dataset à utiliser pour le calcul.

#     Returns:
#         list: Une liste de motifs fréquents.
#     """
#     motif_counts = collections.Counter()
    
#     # 1. Obtenir les indices du subset de données
#     all_indices = dataset.indices
#     print(len(all_indices))
    
#     # 2. Déterminer le nombre d'échantillons et les sélectionner aléatoirement
#     num_to_sample = int(len(all_indices) * sample_percentage)
#     if num_to_sample == 0 and len(all_indices) > 0:
#         num_to_sample = 1
        
#     sampled_indices = rd.sample(all_indices, k=num_to_sample)
    
#     print(f"Sampling {num_to_sample} graphs out of {len(all_indices)} for vocabulary mining.")
    
#     # 3. Parcourir uniquement les graphes de l'échantillon pour trouver les motifs
#     for idx in tqdm(sampled_indices, desc="Mining subgraph motifs from sample"):
#         data = dataset.dataset[idx] # Accéder au graphe via l'index du dataset original
        
#         if data.x is None:
#             continue
        
#         for node_idx in range(data.num_nodes):
#             subset, _, _, _ = k_hop_subgraph(
#                 node_idx, num_hops=1, edge_index=data.edge_index, relabel_nodes=False
#             )
            
#             motif = frozenset(subset.tolist())
#             motif_counts[motif] += 1
            
#     # Extraire les motifs les plus fréquents
#     most_common_motifs = [motif for motif, count in motif_counts.most_common(max_vocab_size)]

#     print(f"\nSubgraph vocabulary found ({len(most_common_motifs)} motifs):")
#     for motif in most_common_motifs:
#         print(f"  - {list(motif)}")
        
#     return most_common_motifs



# def get_subgraph_vocabulary(
#     dataset: data_utils.Subset, 
#     max_vocab_size=30, 
#     sample_percentage=0.4,
#     num_hops=4  # Ajout du paramètre pour la portée de la recherche
# ):
#     """
#     Détermine les motifs de nœuds fréquents à partir d'un échantillon
#     du dataset pour construire un vocabulaire.
    
#     Args:
#         dataset (data_utils.Subset): Le subset d'objets PyTorch Geometric Data.
#         max_vocab_size (int): La taille maximale du vocabulaire de motifs.
#         sample_percentage (float): La proportion du dataset à utiliser pour le calcul.
#         num_hops (int): Le nombre de sauts pour déterminer le voisinage de chaque nœud.

#     Returns:
#         list: Une liste de motifs fréquents (sous forme de frozenset d'indices de nœuds).
#     """
#     motif_counts = collections.Counter()
    
#     all_indices = dataset.indices
#     num_to_sample = int(len(all_indices) * sample_percentage)
#     if num_to_sample == 0 and len(all_indices) > 0:
#         num_to_sample = 1
        
#     sampled_indices = rd.sample(all_indices, k=num_to_sample)
    
#     print(f"Sampling {num_to_sample} graphs out of {len(all_indices)} for vocabulary mining (num_hops={num_hops}).")
    
#     for idx in tqdm(sampled_indices, desc="Mining subgraph motifs from sample"):
#         data = dataset.dataset[idx]
        
#         if data.x is None:
#             continue
        
#         for node_idx in range(data.num_nodes):
#             # Utiliser le nouveau paramètre num_hops
#             subset, _, _, _ = k_hop_subgraph(
#                 node_idx, num_hops=num_hops, edge_index=data.edge_index, relabel_nodes=False
#             )
            
#             # FILTRER les motifs triviaux (ceux qui ne sont que de simples arêtes ou nœuds isolés)
#             if len(subset) > 2:
#                 motif = frozenset(subset.tolist())
#                 motif_counts[motif] += 1
            
#     most_common_motifs = [motif for motif, count in motif_counts.most_common(max_vocab_size)]

#     print(f"\nSubgraph vocabulary found ({len(most_common_motifs)} motifs):")
#     for motif in most_common_motifs:
#         print(f"  - Motif de taille {len(motif)} : {list(motif)}")
        
#     return most_common_motifs
# def get_subgraph_vocabulary(
#     dataset: data_utils.Subset, 
#     max_vocab_size=30, 
#     sample_percentage=0.2,
#     num_hops=3
# ):
#     """
#     Détermine les motifs de nœuds fréquents à partir d'un échantillon
#     du dataset pour construire un vocabulaire, en utilisant les caractéristiques
#     des nœuds pour une représentation canonique.
#     """
#     motif_counts = collections.Counter()
    
#     all_indices = dataset.indices
#     num_to_sample = int(len(all_indices) * sample_percentage)
#     if num_to_sample == 0 and len(all_indices) > 0:
#         num_to_sample = 1
        
#     sampled_indices = rd.sample(all_indices, k=num_to_sample)
    
#     print(f"Sampling {num_to_sample} graphs out of {len(all_indices)} for vocabulary mining (num_hops={num_hops}).")
    
#     for idx in tqdm(sampled_indices, desc="Mining subgraph motifs from sample"):
#         data = dataset.dataset[idx]
        
#         if data.x is None:
#             continue
        
#         for node_idx in range(data.num_nodes):
#             subset, _, _, _ = k_hop_subgraph(
#                 node_idx, num_hops=num_hops, edge_index=data.edge_index, relabel_nodes=False
#             )
            
#             if len(subset) > 2:
#                 # Créer une représentation canonique en utilisant les caractéristiques des nœuds
#                 # 1. Sélectionner les caractéristiques des nœuds du sous-graphe
#                 subgraph_features = data.x[subset]
                
#                 # 2. Trier les caractéristiques pour une représentation unique
#                 #    Nous convertissons le tensor en tuple pour qu'il soit hachable
#                 sorted_features_tuple = tuple(
#                     torch.sort(subgraph_features, dim=0).values.flatten().tolist()
#                 )
                
#                 # Utiliser cette représentation canonique comme motif
#                 motif = frozenset(sorted_features_tuple)
#                 motif_counts[motif] += 1
            
#     most_common_motifs = [motif for motif, count in motif_counts.most_common(max_vocab_size)]

#     print(f"\nSubgraph vocabulary found ({len(most_common_motifs)} motifs):")
#     for motif in most_common_motifs:
#         print(f"  - Motif de taille {len(motif)}: {list(motif)}")
        
#     return most_common_motifs


# def add_hyper_edges_to_dataset(dataset, hyperedge_vocabulary, num_hops=3):
#     """
#     Ajoute des hyperarêtes à chaque graphe en se basant sur la présence de motifs
#     du vocabulaire.
    
#     Args:
#         dataset (list): La liste d'objets PyG Data.
#         hyperedge_vocabulary (list): Une liste de motifs (ensembles de nœuds) à rechercher.
#         num_hops (int): Le nombre de sauts pour trouver les motifs. Doit correspondre
#                         à la valeur utilisée pour créer le vocabulaire.
        
#     Returns:
#         list: Un nouveau dataset avec les hyperarêtes ajoutées.
#     """
#     new_dataset = []
    
#     # Utiliser un ensemble pour une recherche efficace du vocabulaire
#     vocab_set = set(hyperedge_vocabulary)

#     for data in tqdm(dataset, desc="Adding hyper-edges"):
#         new_data = data.clone()
        
#         if not hasattr(new_data, 'hyperedges') or new_data.hyperedges is None:
#             new_data.hyperedges = []
        
#         # Parcourir chaque nœud pour identifier les motifs correspondants
#         for node_idx in range(new_data.num_nodes):
#             # Utiliser la valeur 'num_hops' pour trouver les motifs de la bonne taille
#             subset, _, _, _ = k_hop_subgraph(
#                 node_idx, num_hops=num_hops, edge_index=new_data.edge_index, relabel_nodes=False
#             )
            
#             # Vérifier si le sous-graphe a la taille requise
#             if len(subset) > 2:
#                 # Créer une représentation canonique en utilisant les caractéristiques des nœuds
#                 subgraph_features = data.x[subset]
#                 sorted_features_tuple = tuple(
#                     torch.sort(subgraph_features, dim=0).values.flatten().tolist()
#                 )
                
#                 # Utiliser cette représentation canonique pour la recherche
#                 motif = frozenset(sorted_features_tuple)

#                 # Si le motif est dans le vocabulaire, ajouter une hyperarête
#                 if motif in vocab_set:
#                     # L'hyperarête est l'ensemble des indices de nœuds du sous-graphe
#                     new_hyperedge = list(subset.tolist())
                    
#                     if len(new_hyperedge) > 1: # S'assurer que l'hyperarête a plus d'un nœud
#                         new_data.hyperedges.append(new_hyperedge)
                    
#         new_dataset.append(new_data)
        
#     return new_dataset





# def get_subgraph_vocabulary(dataset, max_subgraph_size=3, max_vocab_size=30):
#     """
#     Mines frequent subgraphs from the dataset to build a vocabulary.
#     Args:
#         dataset (list): A list of PyTorch Geometric Data objects.
#         max_subgraph_size (int): The maximum number of nodes in a subgraph.
#         max_vocab_size (int): The maximum size of the subgraph vocabulary.

#     Returns:
#         list: A list of frequent subgraphs (represented as a matrix of d (number of nodes in this hyperedges) times n (for each of those nodes their corresponding features)).
#         add self-edges to include all possible hyperedges
#     """
#     node_feature_counts = collections.Counter()

#     for data in tqdm(dataset, desc="Counting node feature frequencies"):
#         if data.x is not None: # Check if the graph has node features
#             for node_features in data.x: # Iterate through each node in the graph
#                 # Convert the node features to a hashable tuple and update the count
#                 node_feature_counts[tuple(node_features.tolist())] += 1

#     most_common_nodes = node_feature_counts.most_common(max_vocab_size) # Extract just the features (the first element of each tuple)
#     subgraph_vocabulary = [(list(features), count) for features, count in most_common_nodes]

#     print("\nSubgraph vocabulary (top 30 most frequent features):")
#     print(subgraph_vocabulary)

#     ## not finished
#     ## search among the subgraph vocab which pair (of fragments in the vocab, not necessarily nodes in the futures, might be hyperedges as we union more thing together)
#     # is most frequent and add it to the vocab, repat the operation with the new added vocab in on the loop

#     return subgraph_vocabulary

# def get_subgraph_vocabulary(dataset, max_subgraph_size=3, max_vocab_size=30):
#     """
#     Détermine les motifs de nœuds fréquents (nœud + ses voisins) pour construire un vocabulaire.
#     Ceci est une version simplifiée de la fouille de sous-graphes.
    
#     Args:
#         dataset (list): Une liste d'objets PyTorch Geometric Data.
#         max_subgraph_size (int): La taille maximale du motif (nœud central + voisins).
#         max_vocab_size (int): La taille maximale du vocabulaire de motifs.

#     Returns:
#         list: Une liste de motifs fréquents (sous forme de tuples de caractéristiques de nœuds).
#     """
#     motif_counts = collections.Counter()
    
#     # Parcourir chaque graphe pour trouver les motifs de nœuds
#     for data in tqdm(dataset, desc="Mining subgraph motifs"):
#         if data.x is None:
#             continue
        
#         # Pour chaque nœud, identifier le motif (nœud + ses voisins)
#         for node_idx in range(data.num_nodes):
#             # Utiliser k_hop_subgraph pour trouver les nœuds dans un voisinage
#             # Ici, on prend le voisinage direct (k=1)
#             subset, edge_index, mapping, edge_mask = k_hop_subgraph(
#                 node_idx, num_hops=1, edge_index=data.edge_index, relabel_nodes=False
#             )
            
#             # Créer un motif en utilisant les indices de nœuds triés
#             motif = frozenset(subset.tolist())
            
#             # Compter la fréquence de ce motif
#             motif_counts[motif] += 1
            
#     # Extraire les motifs les plus fréquents
#     most_common_motifs = [motif for motif, count in motif_counts.most_common(max_vocab_size)]

#     print(f"\nSubgraph vocabulary found ({len(most_common_motifs)} motifs):")
#     for motif in most_common_motifs:
#         print(f"  - {list(motif)}")
        
#     return most_common_motifs
# def get_subgraph_vocabulary(dataset, max_subgraph_size=3, max_vocab_size=30, sample_percentage=10.0):
#     """
#     Détermine les motifs de nœuds fréquents (nœud + ses voisins) à partir d'un échantillon
#     du dataset pour construire un vocabulaire.
    
#     Args:
#         dataset (list): Une liste d'objets PyTorch Geometric Data.
#         max_subgraph_size (int): La taille maximale du motif (nœud central + voisins).
#         max_vocab_size (int): La taille maximale du vocabulaire de motifs.
#         sample_percentage (float): La proportion du dataset à utiliser pour le calcul (entre 0.0 et 1.0).

#     Returns:
#         list: Une liste de motifs fréquents (sous forme de frozenset d'indices de nœuds).
#     """
#     motif_counts = collections.Counter()
    
#     # Déterminer la taille de l'échantillon
#     num_to_sample = int(len(dataset) * sample_percentage)
    
#     # S'assurer que le nombre d'échantillons est au moins 1 (sauf si le dataset est vide)
#     if num_to_sample == 0 and len(dataset) > 0:
#         num_to_sample = 1
        
#     # Sélectionner un sous-ensemble aléatoire du dataset
#     # Correction :
#     sampled_dataset = rd.sample(list(dataset), k=num_to_sample)
    
#     # Parcourir chaque graphe de l'échantillon pour trouver les motifs de nœuds
#     for data in tqdm(sampled_dataset, desc="Mining subgraph motifs from sample"):
#         if data.x is None:
#             continue
        
#         # Pour chaque nœud, identifier le motif (nœud + ses voisins)
#         for node_idx in range(data.num_nodes):
#             # Utiliser k_hop_subgraph pour trouver les nœuds dans un voisinage
#             subset, edge_index, mapping, edge_mask = k_hop_subgraph(
#                 node_idx, num_hops=1, edge_index=data.edge_index, relabel_nodes=False
#             )
            
#             # Créer un motif en utilisant les indices de nœuds triés
#             motif = frozenset(subset.tolist())
            
#             # Compter la fréquence de ce motif
#             motif_counts[motif] += 1
            
#     # Extraire les motifs les plus fréquents
#     most_common_motifs = [motif for motif, count in motif_counts.most_common(max_vocab_size)]

#     print(f"\nSubgraph vocabulary found ({len(most_common_motifs)} motifs) from a sample of {num_to_sample} graphs:")
#     for motif in most_common_motifs:
#         print(f"  - {list(motif)}")
        
#     return most_common_motifs