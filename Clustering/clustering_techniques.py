import numpy as np
import pandas as pd
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.fcm import fcm
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift, Birch, SpectralClustering, SpectralCoclustering

from MDE import Concepts
from Utils import metrics

# kmeans
def kmeans(X, k, files_name):
    print("------------------------Kmeans---------------------")
    model = KMeans(n_clusters=k, init='k-means++', max_iter=200, n_init=10)
    model.fit(X)
    labels = model.labels_
    source_cl = pd.DataFrame(list(zip(files_name, labels)), columns=['file name', 'cluster'])
    print(source_cl.sort_values(by=['cluster']).to_string())
    predicted_concepts = Concepts.extract_concepts(source_cl)

    #plot_clusters(X, k, files_name)
    return labels, predicted_concepts


# Dbscan
def dbscan(X, files_name, dataset_type):
    print("------------------------DBScan---------------------")
    # Compute DBSCAN
    db = DBSCAN(eps=0.8, min_samples=2).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    # plot_clusters(X, labels, files_name, None, dim_reducer='svd')

    source_cl = pd.DataFrame(list(zip(files_name, labels)), columns=['file name', 'cluster'])
    print(source_cl.sort_values(by=['cluster']).to_string())
    predicted_concepts = Concepts.extract_concepts(source_cl)

    return labels, predicted_concepts


def agglomerative(X, k, files_name):
    print("------------------------Agglomerative Clustering---------------------")
    model = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
    model.fit(X.toarray())
    labels = model.labels_
    source_cl = pd.DataFrame(list(zip(files_name, labels)), columns=['file name', 'cluster'])
    print(source_cl.sort_values(by=['cluster']).to_string())
    predicted_concepts = Concepts.extract_concepts(source_cl)

    return labels, predicted_concepts


def mean_shift(X, files_name):
    print("------------------------Mean Shift Clustering---------------------")
    model = MeanShift()
    model.fit(X.toarray())
    labels = model.labels_
    source_cl = pd.DataFrame(list(zip(files_name, labels)), columns=['file name', 'cluster'])
    # print(source_cl.sort_values(by=['cluster']).to_string())
    Concepts.extract_concepts(source_cl)
    print(metrics.davies_bouldin_score(X, labels))


def birch(X, files_name):
    print("------------------------Birch Clustering---------------------")
    model = Birch(n_clusters=None)
    model.fit(X)
    labels = model.labels_
    source_cl = pd.DataFrame(list(zip(files_name, labels)), columns=['file name', 'cluster'])
    print(source_cl.sort_values(by=['cluster']).to_string())

    predicted_concepts = Concepts.extract_concepts(source_cl)
    # plot_clusters(X, k, files_name)
    return labels, predicted_concepts


def spectral_clustering(X, k, files_name):
    print("------------------------Spectral Clustering---------------------")
    model = SpectralClustering(n_clusters=k, assign_labels='discretize', random_state=0)
    model.fit(X)
    labels = model.labels_
    source_cl = pd.DataFrame(list(zip(files_name, labels)), columns=['file name', 'cluster'])
    print(source_cl.sort_values(by=['cluster']).to_string())
    predicted_concepts = Concepts.extract_concepts(source_cl)

    return labels, predicted_concepts


def spectral_co_clustering(X, k, files_name):
    print("------------------------Spectral Co Clustering---------------------")
    model = SpectralCoclustering(n_clusters=k, random_state=0)
    model.fit(X)
    labels = model.row_labels_
    source_cl = pd.DataFrame(list(zip(files_name, labels)), columns=['file name', 'cluster'])
    print(source_cl.sort_values(by=['cluster']).to_string())
    Concepts.extract_concepts(source_cl)


def fuzzy_cmeans(X, files_name):
    print("------------------------Fuzzy C-Means---------------------")
    X_arr = X.toarray()
    initial_centers = kmeans_plusplus_initializer(X_arr, 3, kmeans_plusplus_initializer.FARTHEST_CENTER_CANDIDATE).initialize()

    # create instance of Fuzzy C-Means algorithm
    fcm_instance = fcm(X_arr, initial_centers)

    # run cluster analysis and obtain results
    fcm_instance.process()
    clusters = fcm_instance.get_clusters()
    centers = fcm_instance.get_centers()

    print(fcm_instance.get_membership())
    # labels = centers
    # source_cl = pd.DataFrame(list(zip(files_name, labels)), columns=['file name', 'cluster'])
    # print(source_cl.sort_values(by=['cluster']).to_string())
    # Concepts.extract_concepts(source_cl)

    # visualize clustering results
    # visualizer = cluster_visualizer()
    # visualizer.append_clusters(clusters, X_arr)
    # visualizer.append_cluster(centers, marker='*', markersize=10)
    # visualizer.show()


# def clique(x):
#     # https://pyclustering.github.io/docs/0.9.0/html/d2/d4f/classpyclustering_1_1cluster_1_1clique_1_1clique.html
