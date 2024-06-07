
import random

from typing import List, Dict

import numpy as np
import random

from sklearn.cluster import KMeans
from alternative_image_clustering.metrics import get_metrics

RANDOM_STATE = 42


def run_single_kmeans(data: np.ndarray, n_clusters: int, random_state: int):

    parameters = {
        "n_clusters": n_clusters,
        "init": "k-means++",
        "n_init": 1,
        "random_state": random_state,
    }
    model = KMeans(**parameters)
    clustering = model.fit(data)

    return {
        "labels": clustering.labels_,
        "inertia": model.inertia_,
        "random_state": random_state,
    }


def kmeans(data: np.ndarray, labels: np.ndarray, n_clusters: int, num_runs=10):
    """
    Perform K-means clustering on the data.

    Parameters
    ----------
    data : np.ndarray
        The data to cluster.
    labels : Dict[np.ndarray
        The labels to compare against.
    n_clusters : int
        The number of clusters to use.
    num_runs : int
        The number of runs to perform.
    """
    random.seed(RANDOM_STATE)
    random_states = random.sample(range(1000), num_runs)

    runs = []
    for random_state in random_states:

        run = run_single_kmeans(
            data=data, n_clusters=n_clusters, random_state=random_state
        )
        if labels is not None:
            metrics = get_metrics(labels_true=labels, labels_pred=run["labels"])
        else:
            metrics = None

        run["metrics"] = metrics
        runs.append(run)

    best_run = min(runs, key=lambda x: x["inertia"])
    info = {
        "best_inertia": best_run["inertia"],
        "best_labels": best_run["labels"],
        "best_random_state": best_run["random_state"],
    }
    if labels is not None:
        info["metrics"] = {
            metric: np.mean([run["metrics"][metric] for run in runs])
            for metric in runs[0]["metrics"]
        }
        info["metrics_stddev"] = {
            metric: np.std([run["metrics"][metric] for run in runs])
            for metric in runs[0]["metrics"]
        }

    return info