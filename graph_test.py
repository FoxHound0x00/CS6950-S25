import numpy as np
import networkx as nx
from scipy.spatial import KDTree
from sklearn.metrics import pairwise_distances

def vr_relative_neighborhood_graph(X, k, epsilon):
    """
    Constructs a VR-relative neighborhood graph for a given point cloud X.
    
    Parameters:
    X : ndarray of shape (n, d)
        The input point cloud, where n is the number of points and d is the dimension.
    k : int
        The fixed integer determining the k-th nearest neighbor.
    epsilon : float
        The scale parameter controlling connectivity.
    
    Returns:
    G : networkx.Graph
        The constructed VR-relative neighborhood graph.
    """
    n, d = X.shape
    tree = KDTree(X)
    G = nx.Graph()
    
    # Add nodes
    for i in range(n):
        G.add_node(i, pos=X[i])
    
    # Compute k-th nearest neighbor distances
    distances, indices = tree.query(X, k=k+1)  # k+1 first nn is itself 

    for i in range(n):
        d_xk = distances[i, k]  # k-th nearest neighbor within distance of x
        for j in range(i + 1, n):
            d_yk = distances[j, k]  # k-th nearest neighbor within distance of y
            d_xy = np.linalg.norm(X[i] - X[j])  # Euclidean distance between x and y
            dist = d_xy / max(d_xk, d_yk) 
            
            if dist <= epsilon: 
                G.add_edge(i, j, weight=dist)
    
    return G, distances, indices
