import multiprocessing
import ClusterEnsembles as CE
import numpy as np
import random

from typing import List, Dict

from prometheus_client import Info
from sklearn.cluster import KMeans
from clustpy.alternative import NrKmeans
from tqdm import tqdm

from alternative_image_clustering.metrics import (
    get_metrics,
    get_multiple_labeling_metrics,
)

RANDOM_STATE = 812


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

    # cache_name = f"kmeans{n_clusters}_{n_iterations}"
    # cached_results = _load_from_disk(data, cache_name)
    # if cached_results is not None:
    #     # print("Clustering loaded from disk.")
    #     if not return_inertia:
    #         return cached_results

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

    # if cached_results is None:
    #     _save_to_disk(data, cache_name, clustering.labels_)
    return info


def run_single_nr_kmeans(data, labels, n_clusters, random_state):
    """Function to be executed by each worker process."""

    categories = list(n_clusters.keys())

    parameters = {
        "n_clusters": list(n_clusters.values()),
        "max_iter": 300,
        "random_state": random_state,
        "debug": True,
    }
    model = NrKmeans(**parameters).fit(data)
    pred_clusterings = model.labels_

    return {
        "labels": pred_clusterings,
        "cost": model.calculate_cost_function(),
        "metrics": get_multiple_labeling_metrics(
            labels_true=labels,
            labels_pred=pred_clusterings,
            categories=categories,
        ),
        "random_state": random_state,
    }


def nrkmeans(
    data: np.ndarray,
    labels: np.ndarray,
    n_clusters: Dict[str, int],
    num_runs=10,
):

    random.seed(RANDOM_STATE)
    random_states = random.sample(range(1000), num_runs)

    categories = list(n_clusters.keys())
    parameters = [
        (data, labels, n_clusters, random_state) for random_state in random_states
    ]

    # Multiprocessing with a Pool of workers
    with multiprocessing.Pool() as pool:
        runs = list(
            tqdm(pool.starmap(run_single_nr_kmeans, parameters), total=num_runs)
        )

    best_run = min(runs, key=lambda x: x["cost"])
    info = {
        "best_cost": best_run["cost"],
        "best_labels": best_run["labels"],
        "best_random_state": best_run["random_state"],
    }
    info["metrics"] = {
        category: {
            metric: np.mean([run["metrics"][category][metric] for run in runs])
            for metric in runs[0]["metrics"][category]
        }
        for category in categories
    }
    info["metrics_stddev"] = {
        category: {
            metric: np.std([run["metrics"][category][metric] for run in runs])
            for metric in runs[0]["metrics"][category]
        }
        for category in categories
    }
    return info


def consensus_clustering(labels: np.ndarray):
    if labels.shape[0] == 1:
        return labels.flatten()
    labels_ce = CE.cluster_ensembles(labels, solver="mcla", random_state=RANDOM_STATE)
    return labels_ce
