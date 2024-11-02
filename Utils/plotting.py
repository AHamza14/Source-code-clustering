import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.manifold import MDS

from sklearn.metrics.pairwise import cosine_similarity
# % matplotlib inline


import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, PCA


def plot_clusters(X, y, file_names, cluster_centers=None, dim_reducer='svd'):
    """
    Plots the clusters after applying a clustering technique.

    Parameters:
    - X: Feature matrix (can be sparse or dense).
    - y: Cluster labels (output from the clustering technique).
    - file_names: List of file names corresponding to rows in X.
    - cluster_centers: Centers of the clusters (optional, if provided by the clustering algorithm: kmeans and birch).
    - dim_reducer: Dimensionality reduction technique ('pca' or 'svd').
    """

    # Reduce dimensionality to 2D
    if dim_reducer == 'pca':
        reducer = PCA(n_components=2)
    else:  # Default to TruncatedSVD if PCA is not selected
        reducer = TruncatedSVD(n_components=2)

    X_reduced = reducer.fit_transform(X)

    # Plot clusters
    plt.figure(figsize=(10, 7))

    # Create a scatter plot with colors representing cluster assignments
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis', s=50, alpha=0.6)

    # Add the file names as annotations
    for i, file_name in enumerate(file_names):
        plt.annotate(file_name, (X_reduced[i, 0], X_reduced[i, 1]), fontsize=8, alpha=0.75)

    # Add the cluster centers if provided
    if cluster_centers is not None:
        centers_reduced = reducer.transform(cluster_centers)
        plt.scatter(centers_reduced[:, 0], centers_reduced[:, 1], c='red', s=300, marker='x', label='Cluster Centers')

    plt.title('Clustering of Source Code Files')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.grid(True)
    plt.show()
