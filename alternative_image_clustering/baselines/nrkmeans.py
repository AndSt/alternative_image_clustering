import multiprocessing
import numpy as np
import random

import random
from typing import List, Dict

from alternative_image_clustering.clustering import RANDOM_STATE
from clustpy.alternative import NrKmeans

from tqdm import tqdm

from alternative_image_clustering.metrics import get_multiple_labeling_metrics

RANDOM_STATE = 812

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