import os
import pandas as pd


def export_metrics_to_excel(clustering_technique, dataset_path, dataset_type, prepro_scen, num_clusters=None,
                            truth_concepts=None, discovered_concepts=None, num_concepts=None,
                            silhouette=None, calinski_harabasz=None, davies_bouldin=None,
                            true_positive=None, false_positive=None, false_negative=None,
                            precision=None, recall=None, f_score=None):

    discovered_cons = ", ".join([item.strip() for item in discovered_concepts])
    truth_cons = ", ".join([item.strip() for item in truth_concepts])

    metrics_values = {'Number of clusters': num_clusters,
        'Ground truth Concepts': truth_cons, 'Predicted Concepts': discovered_cons, 'Number of predicted Concepts': num_concepts,
        'Silhouette': silhouette, 'Calinski Harabasz': calinski_harabasz,
        'Davies Bouldin': davies_bouldin, 'True positive': true_positive, 'False positive': false_positive,
        'False negative': false_negative, 'Precision (%)': precision, 'Recall (%)': recall, 'F-Score (%)': f_score}

    # Filter out any None values
    metrics_values = {k: v for k, v in metrics_values.items() if v is not None}

    # Convert the metrics dictionary to a DataFrame
    df = pd.DataFrame(list(metrics_values.items()), columns=['Metrics', 'Values'])

    # Create the folder if it doesn't exist
    folder_name = "Results"
    os.makedirs(folder_name, exist_ok=True)

    # Combine folder path and filename
    dataset_name = os.path.basename(dataset_path)
    filename = dataset_name.capitalize() + "_" + dataset_type.capitalize() + "-dataset_" + clustering_technique + "_" +prepro_scen + ".xlsx"
    file_path = os.path.join(folder_name, filename)
    # Export to Excel

    df.to_excel(file_path, sheet_name='Metrics', index=False)
    print(f"Metrics exported to {file_path} successfully.")

