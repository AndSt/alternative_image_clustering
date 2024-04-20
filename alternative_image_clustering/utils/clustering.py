from hashlib import sha256
from os.path import exists

import ClusterEnsembles as CE
import numpy as np

import warnings

# warnings.simplefilter(action="ignore", category=FutureWarning)

from clustpy.alternative import NrKmeans
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
    calinski_harabasz_score,
)
from clustpy.metrics.clustering_metrics import unsupervised_clustering_accuracy
from clustpy.partition import XMeans, GMeans

# from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 812


def get_clustering_method(name: str):
    methods = {"kmeans": kmeans, "xmeans": xmeans, "gmeans": gmeans}
    assert name in methods, "Clustering method not found."
    return methods[name]


def kmeans(data, n_clusters, n_iterations=10, return_inertia=False):
    if n_clusters < 1:
        return xmeans(data)

    cache_name = f"kmeans{n_clusters}_{n_iterations}"
    cached_results = _load_from_disk(data, cache_name)
    if cached_results is not None:
        # print("Clustering loaded from disk.")
        if not return_inertia:
            return cached_results

    parameters = {
        "n_clusters": n_clusters,
        "init": "k-means++",
        "n_init": n_iterations,
        "random_state": RANDOM_STATE,
    }
    model = KMeans(**parameters)
    clustering = model.fit(data)

    if cached_results is None:
        _save_to_disk(data, cache_name, clustering.labels_)

    if return_inertia:
        return clustering.labels_, model.inertia_

    return clustering.labels_


def xmeans_(data, max_n_clusters=8, n_iterations=10):
    clusterings = [
        kmeans(data, n_clusters, n_iterations)
        for n_clusters in range(2, max_n_clusters + 1)
    ]
    clustering_best = max(clusterings, key=lambda x: calinski_harabasz_score(data, x))
    return clustering_best


def xmeans(data, max_n_clusters=8, n_iterations=10):
    data_reduced = np.array(data)
    # if data.shape[1] > 50:
    #     data_reduced = PCA(50).fit_transform(data)
    # data_reduced = StandardScaler().fit_transform(data)

    # cached_results = _load_from_disk(data, "xmeans")
    # if cached_results is not None:
    # print("Clustering loaded from disk.")
    # return cached_results

    parameters = {
        "n_clusters_init": 2,
        "max_n_clusters": max_n_clusters,
        "check_global_score": True,
        "allow_merging": False,
        "random_state": RANDOM_STATE,
    }
    clusterings = [XMeans(**parameters).fit(data_reduced) for _ in range(n_iterations)]
    clustering_best = max(
        clusterings, key=lambda x: silhouette_score(data_reduced, x.labels_)
    )

    # _save_to_disk(data, "xmeans", clustering_best.labels_)
    return clustering_best.labels_


def gmeans(data, max_n_clusters=6, n_iterations=10):
    cached_results = _load_from_disk(data, "gmeans")
    if cached_results is not None:
        # print("Clustering loaded from disk.")
        return cached_results

    parameters = {
        "n_clusters_init": 1,
        "max_n_clusters": max_n_clusters,
        "random_state": RANDOM_STATE,
    }

    clusterings = [GMeans(**parameters).fit(data) for _ in range(n_iterations)]
    clustering_best = max(clusterings, key=lambda x: silhouette_score(data, x.labels_))

    _save_to_disk(data, "gmeans", clustering_best.labels_)
    return clustering_best.labels_


def nrkmeans(data, n_clusters):
    # max_dims = min(data.shape[0], 350)
    # if data.shape[1] > max_dims:
    #     data_reduced = PCA(max_dims, random_state=RANDOM_STATE).fit_transform(data)

    model = NrKmeans(n_clusters=n_clusters, max_iter=300, random_state=RANDOM_STATE)

    return model.fit_predict(data)


def ami(labels_true, labels_pred):
    return adjusted_mutual_info_score(labels_true, labels_pred)


def ari(labels_true, labels_pred):
    return adjusted_rand_score(labels_true, labels_pred)


def acc(labels_true, labels_pred):
    return unsupervised_clustering_accuracy(
        np.array(labels_true), np.array(labels_pred)
    )


def get_metrics(labels_true, labels_pred):
    metrics = {
        "AMI": ami(labels_true, labels_pred),
        "ARI": ari(labels_true, labels_pred),
        "ACC": acc(labels_true, labels_pred),
    }
    return metrics


def silhouette(data, labels):
    return silhouette_score(data, labels)


def tsne(data, *args, **kwargs):
    data_reduced = data
    if data.shape[1] > 50:
        data_reduced = PCA(50).fit_transform(data)

    return TSNE(*args, **kwargs).fit_transform(data_reduced)


def consensus_clustering(labels: np.ndarray):
    if labels.shape[0] == 1:
        return labels.flatten()
    labels_ce = CE.cluster_ensembles(labels, solver="mcla", random_state=RANDOM_STATE)
    return labels_ce


def _hash_data(data):
    return sha256(np.ascontiguousarray(data)).hexdigest()


def _save_to_disk(data, method, clustering):
    hash = _hash_data(data)
    file_name = f"clustering_cache/{method}_{hash}.npy"

    np.save(file_name, clustering)


def _load_from_disk(data, method):
    hash = _hash_data(data)
    file_name = f"clustering_cache/{method}_{hash}.npy"

    if not exists(file_name):
        return None

    return np.load(file_name)
