from sklearn import metrics


def internal_metrics(X, labels):
    sil = silhouette_score(X, labels, "euclidean")
    cal = calinski_harabasz_score(X, labels)
    dav = davies_bouldin_score(X, labels)
    return sil, cal, dav


def silhouette_score(X, labels, distance_metric):
    try:
        score = metrics.silhouette_score(X, labels, metric=distance_metric)
        return score
    except Exception as e:
        return "Error: Only one cluster found. At least two are needed for evaluation metric."


def calinski_harabasz_score(X, labels):
    try:
        score = metrics.calinski_harabasz_score(X.toarray(), labels)
        return score
    except Exception as e:
        return "Error: Only one cluster found. At least two are needed for evaluation metric."


def davies_bouldin_score(X, labels):
    try:
        score = metrics.davies_bouldin_score(X.toarray(), labels)
        return score
    except Exception as e:
        return "Error: Only one cluster found. At least two are needed for evaluation metric."


def external_metrics(truth_concepts, predicted_concepts):
    # Convert both lists to sets for easier comparison

    truth_set = set(concept.lower() for concept in truth_concepts)
    predicted_set = set(concept.lower() for concept in predicted_concepts)

    # Calculate true positives (TP), false positives (FP), and false negatives (FN)
    tp = len(truth_set.intersection(predicted_set))
    fp = len(predicted_set.difference(truth_set))
    fn = len(truth_set.difference(predicted_set))

    # Calculate precision, recall, and F1 score
    precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
    recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return tp, fp, fn, precision, recall, f1_score