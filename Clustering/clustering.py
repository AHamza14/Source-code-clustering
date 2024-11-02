from tkinter import messagebox

from Clustering.clustering_techniques import *
from Utils.ML_helpers import *
from Utils.metrics import external_metrics, internal_metrics
from Utils.exporters import export_metrics_to_excel


def cluster_classes(solution_path, type_dataset="normal", type_preprocessing="WoPP", k_clusters=1, truth_concepts = {}):
    print("\n--------------------------------------------------------")
    print(type_dataset, " Dataset with ", type_preprocessing, " preprocessing and k=", k_clusters)

    X, files_name = prepare_dataset_X(solution_path, type_dataset, type_preprocessing)

    # Kmeans
    labels, predicted_concepts = kmeans(X, k_clusters, files_name)

    sil, cal, dav = internal_metrics(X, labels)
    tp, fp, fn, precision, recall, f1_score = external_metrics(truth_concepts, predicted_concepts)
    export_metrics_to_excel(clustering_technique="K-means", dataset_path=solution_path, dataset_type=type_dataset, prepro_scen=type_preprocessing,
                            num_clusters=k_clusters, num_concepts=len(predicted_concepts), discovered_concepts=predicted_concepts, truth_concepts=truth_concepts,
                            silhouette=sil, calinski_harabasz=cal, davies_bouldin=dav,
                            true_positive=tp, false_positive=fp, false_negative=fn,
                            precision=precision, recall=recall, f_score=f1_score)

    # Agglomerative clustering
    labels, predicted_concepts = agglomerative(X, k_clusters, files_name)

    sil, cal, dav = internal_metrics(X, labels)
    tp, fp, fn, precision, recall, f1_score = external_metrics(truth_concepts, predicted_concepts)

    export_metrics_to_excel(clustering_technique="Agglomerative", dataset_path=solution_path, dataset_type=type_dataset,
                            prepro_scen=type_preprocessing, num_clusters=k_clusters,
                            num_concepts=len(predicted_concepts), discovered_concepts=predicted_concepts,
                            truth_concepts=truth_concepts, silhouette=sil, calinski_harabasz=cal, davies_bouldin=dav,
                            true_positive=tp, false_positive=fp, false_negative=fn, precision=precision, recall=recall,
                            f_score=f1_score)
    # # Birch clustering
    labels, predicted_concepts = birch(X, files_name)

    sil, cal, dav = internal_metrics(X, labels)
    tp, fp, fn, precision, recall, f1_score = external_metrics(truth_concepts, predicted_concepts)

    export_metrics_to_excel(clustering_technique="BIRCH", dataset_path=solution_path, dataset_type=type_dataset,
                            prepro_scen=type_preprocessing, num_clusters=k_clusters,
                            num_concepts=len(predicted_concepts), discovered_concepts=predicted_concepts,
                            truth_concepts=truth_concepts, silhouette=sil, calinski_harabasz=cal, davies_bouldin=dav,
                            true_positive=tp, false_positive=fp, false_negative=fn, precision=precision, recall=recall,
                            f_score=f1_score)

    # #DBCScan
    labels, predicted_concepts = dbscan(X, files_name, type_dataset)

    sil, cal, dav = internal_metrics(X, labels)
    tp, fp, fn, precision, recall, f1_score = external_metrics(truth_concepts, predicted_concepts)

    export_metrics_to_excel(clustering_technique="DBSCAN", dataset_path=solution_path, dataset_type=type_dataset,
                            prepro_scen=type_preprocessing, num_clusters=k_clusters,
                            num_concepts=len(predicted_concepts), discovered_concepts=predicted_concepts,
                            truth_concepts=truth_concepts, silhouette=sil, calinski_harabasz=cal, davies_bouldin=dav,
                            true_positive=tp, false_positive=fp, false_negative=fn, precision=precision, recall=recall,
                            f_score=f1_score)

    # Spectral clustering
    labels, predicted_concepts = spectral_clustering(X, k_clusters, files_name)

    sil, cal, dav = internal_metrics(X, labels)
    tp, fp, fn, precision, recall, f1_score = external_metrics(truth_concepts, predicted_concepts)

    export_metrics_to_excel(clustering_technique="Spectral", dataset_path=solution_path, dataset_type=type_dataset,
                            prepro_scen=type_preprocessing, num_clusters=k_clusters,
                            num_concepts=len(predicted_concepts), discovered_concepts=predicted_concepts,
                            truth_concepts=truth_concepts, silhouette=sil, calinski_harabasz=cal, davies_bouldin=dav,
                            true_positive=tp, false_positive=fp, false_negative=fn, precision=precision, recall=recall,
                            f_score=f1_score)

    messagebox.showinfo("Clustering Complete", "Source code clustering has finished, check the Results folder !")


def find_k(test_k, solution_path, type_dataset="normal", type_preprocessing="WoPP"):
    print("\n Finding optimal K for test K:", test_k, "...")
    print(type_dataset, " Dataset with ", type_preprocessing, " preprocessing")

    X, files_name = prepare_dataset_X(solution_path, type_dataset, type_preprocessing)

    # Elbow method
    # elbow_method(X, test_k)

    return silhouette_method(X, test_k)


def prepare_dataset_X(solution_path, type_dataset, type_preprocessing):
    files_name, data = Load_data_set(solution_path)
    print("Number of files (classifiers): ", len(files_name))

    # Data preprocessing
    if type_preprocessing != "WoPP":
        data = data_preprocessing(data, type_preprocessing)

    # add noise
    if type_dataset == "noisy":
        data = data_noise(data)

    # Data vectorization
    X = data_vectorization(data)
    return X, files_name


# def cluster_embeddings(solutions_path):
#     files_name, source_code_data = Load_data_set(solutions_path)
#     # print(source_code_data[6])
#
#     # Data preprocessing
#     cleaned_data = data_preprocessing(source_code_data)
#     # print(cleaned_data[6])
#
#     corpus = build_corpus(cleaned_data)
#     # corpus = cleaned_data
#     # print(corpus)
#
#     # Train Word2Vec model
#     # Example sentences
#
#     model = Word2Vec(corpus, vector_size=100, window=5, min_count=1, workers=4)
#
#     # Access the word vectors
#     word_vectors = model.wv
#
#     ####### PCA
#
#     # Extract word vectors for the words in the vocabulary
#     word_vecs = [word_vectors[word] for word in word_vectors.index_to_key]
#
#     # Perform PCA to reduce dimensions to 2D
#     pca = PCA(n_components=2)
#     result = pca.fit_transform(word_vecs)
#
#     # clustering techniques
#     test_k = 50
#     X = result
#     # Kmeans
#     kmeans(X, test_k, files_name)
