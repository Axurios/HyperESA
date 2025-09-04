import numpy as np
import pandas as pd
import os
import torch
import collections
# import torch_geometric
# from torch_geometric.utils import degree, dense_to_sparse
# from torch_geometric.data import Data, InMemoryDataset
# from torch.utils.data import random_split
# from sklearn.preprocessing import StandardScaler

from tqdm.auto import tqdm
import networkx as nx
# from gspan_mining.gspan import gSpan


# def vocab_in_graph2(vocab, node_set):
#     if vocab.shape[1] != node_set.shape[1]:
#         print("Error: The number of features (columns) must be the same for both matrices.")
#         return False
            
#     m_rows_set = {tuple(row.tolist()) for row in node_set}
#     for row_d in vocab:
#         if tuple(row_d.tolist()) not in m_rows_set:
#             return False
#     return True

def vocab_in_graph(vocab, node_set):
    if vocab.shape[1] != node_set.shape[1]:
        print("Error: The number of features (columns) must be the same for both matrices.")
        return False
    
    if vocab.size == 0:
        return True
    
    vocab_sorted = np.ascontiguousarray(vocab).view([('', vocab.dtype)] * vocab.shape[1])
    node_set_sorted = np.ascontiguousarray(node_set).view([('', node_set.dtype)] * node_set.shape[1])
    
    vocab_sorted.sort(order=vocab_sorted.dtype.names)
    node_set_sorted.sort(order=node_set_sorted.dtype.names)
    
    is_in_set = np.isin(vocab_sorted, node_set_sorted)
    return np.all(is_in_set)



def get_subgraph_vocabulary(dataset, max_subgraph_size=3, max_vocab_size=30):
    """
    Mines frequent subgraphs from the dataset to build a vocabulary.
    Args:
        dataset (list): A list of PyTorch Geometric Data objects.
        max_subgraph_size (int): The maximum number of nodes in a subgraph.
        max_vocab_size (int): The maximum size of the subgraph vocabulary.

    Returns:
        list: A list of frequent subgraphs (represented as a matrix of d (number of nodes in this hyperedges) times n (for each of those nodes their corresponding features)).
        add self-edges to include all possible hyperedges
    """
    node_feature_counts = collections.Counter()

    for data in tqdm(dataset, desc="Counting node feature frequencies"):
        if data.x is not None: # Check if the graph has node features
            for node_features in data.x: # Iterate through each node in the graph
                # Convert the node features to a hashable tuple and update the count
                node_feature_counts[tuple(node_features.tolist())] += 1

    most_common_nodes = node_feature_counts.most_common(max_vocab_size) # Extract just the features (the first element of each tuple)
    subgraph_vocabulary = [(list(features), count) for features, count in most_common_nodes]

    print("\nSubgraph vocabulary (top 30 most frequent features):")
    print(subgraph_vocabulary)

    ## not finished

    return subgraph_vocabulary





def add_hyper_edges_to_dataset(dataset, hyperedge_vocabulary):
    pass
    # for each graph, find if there is any subgrap that is in the vocab
    new_dataset = []
    for data in tqdm(dataset, desc="Adding hyper-edges"):
        new_data = data.clone()

        original_edge_index = new_data.edge_index ; original_edge_attr = new_data.edge_attr
        hyper_edge_index = [] ; hyper_edge_attr = []
        node_set = []
        
        #if found, add it to the edge_index and add also its edge_attributes
        for vocab in hyperedge_vocabulary:
            is_vocab_found_in_this_graph = vocab_in_graph(vocab, node_set)
            if is_vocab_found_in_this_graph:
                pass
        
        # If any hyper-edges were found, add them to the graph
        if hyper_edge_index:
            pass
            
        new_dataset.append(new_data)
    return new_dataset # the new dataset with the added hyperedges