from itertools import permutations

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
    normalized_mutual_info_score,
)
from clustpy.metrics.clustering_metrics import unsupervised_clustering_accuracy
from clustpy.metrics.multipe_labelings_scoring import MultipleLabelingsConfusionMatrix


def get_metrics(labels_true, labels_pred):
    metrics = {
        "AMI": adjusted_mutual_info_score(labels_true, labels_pred),
        "NMI": normalized_mutual_info_score(labels_true, labels_pred),
        "ARI": adjusted_rand_score(labels_true, labels_pred),
        "ACC": unsupervised_clustering_accuracy(
            np.array(labels_true), np.array(labels_pred)
        ),
        # "Silhouette": silhouette_score(labels_true, labels_pred),
    }
    return metrics


def get_best_assignment_metrics(matrix, categories):
    all_permutations = list(permutations(range(len(categories)), len(categories)))

    scores = {}
    for permutation in all_permutations:
        score = 0
        for i in range(3):
            score += matrix[permutation[i], i]
        scores[permutation] = score

    best_permutation = max(scores, key=scores.get)

    return {
        category: matrix[best_permutation[idx], idx]
        for idx, category in enumerate(categories)
    }


def get_multiple_labeling_metrics(label_true, labels_pred, categories):

    metric_name_func = {
        "AMI": adjusted_mutual_info_score,
        "NMI": normalized_mutual_info_score,
        "ARI": adjusted_rand_score,
        "ACC": unsupervised_clustering_accuracy,
    }

    metrics = {category: {} for category in categories}

    for metric_name, metric_func in metric_name_func.items():
        matrix = MultipleLabelingsConfusionMatrix(
            labels_true=label_true, labels_pred=labels_pred, metric=metric_func
        ).rearrange_matrix()
        metric_values = get_best_assignment_metrics(matrix.confusion_matrix, categories)
        for category in categories:
            metrics[category][metric_name] = metric_values[category]

    return metrics
