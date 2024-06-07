import numpy as np
import copy

import os

from alternative_image_clustering.data.dataset import Dataset
from alternative_image_clustering.clustering import kmeans
from alternative_image_clustering.baselines.nrkmeans import nrkmeans
from alternative_image_clustering.baselines.enrc import enrc
from alternative_image_clustering.baselines.orth_run import orth



def run_full_kmeans(dataset: str, embedding_type: str, base_dir: str):

    # load data
    dataset: Dataset = Dataset(
        base_dir=base_dir,
        dataset_name=dataset,
        embedding_type=embedding_type,
    )

    # print("Running kmeans on full embeddings for each category")
    full_kmeans_performance = {}
    embeddings = dataset.get_embeddings()

    for category in dataset.get_categories():
        labels = dataset.get_clustering_labels(category)
        n_clusters = len(set(labels))
        kmeans_result = kmeans(embeddings, labels, n_clusters=n_clusters)
        full_kmeans_performance[category] = kmeans_result

    # save the results

    return full_kmeans_performance


def run_per_prompt_kmeans(dataset: str, embedding_type: str, base_dir: str):
    # load data
    dataset: Dataset = Dataset(
        base_dir=base_dir,
        dataset_name=dataset,
        embedding_type=embedding_type,
    )

    # go over all prompts
    # print("Running kmeans for each prompt")
    per_prompt_performance = {}
    for idx, prompt_info in dataset.prompt_dict.items():
        dataset.active_prompt_indices = [idx]
        embeddings = dataset.get_embeddings()
        labels = dataset.get_clustering_labels(prompt_info["category"])
        kmeans_result = kmeans(embeddings, labels, n_clusters=len(set(labels)))
        per_prompt_performance[idx] = kmeans_result

    # compute per category performance
    # print("Transforming to average and max. performance")

    metric_list = list(per_prompt_performance[list(per_prompt_performance.keys())[0]]["metrics"].keys())
    metric_list = {metric: [] for metric in metric_list}
    empty_metrics_dict = {category: copy.deepcopy(metric_list) for category in dataset.get_categories()}

    results = {
        "per_category_avg_performance": copy.deepcopy(empty_metrics_dict),
        "per_category_avg_performance_stddev": copy.deepcopy(empty_metrics_dict),
        "per_category_max_performance": copy.deepcopy(empty_metrics_dict),
        "per_category_max_performance_stddev": copy.deepcopy(empty_metrics_dict),
    }

    for idx, info in per_prompt_performance.items():

        category = dataset.prompt_dict[idx]["category"]

        for metric in info["metrics"]:

            results["per_category_avg_performance"][category][metric].append(
                info["metrics"][metric]
            )
            results["per_category_avg_performance_stddev"][category][metric].append(
                info["metrics_stddev"][metric]
            )

            results["per_category_max_performance"][category][metric].append(
                info["metrics"][metric]
            )
            results["per_category_max_performance_stddev"][category][metric].append(
                info["metrics_stddev"][metric]
            )

    for category in results["per_category_avg_performance"]:
        for metric in results["per_category_avg_performance"][category]:

            results["per_category_avg_performance"][category][metric] = np.mean(
                results["per_category_avg_performance"][category][metric]
            )
            results["per_category_avg_performance_stddev"][category][metric] = np.mean(results["per_category_avg_performance_stddev"][category][metric])

            max_id = np.argmax(results["per_category_max_performance"][category][metric])
            results["per_category_max_performance"][category][metric] = results["per_category_max_performance"][category][metric][max_id]
            results["per_category_max_performance_stddev"][category][metric] = results["per_category_max_performance_stddev"][category][metric][max_id]

    results["per_prompt_performance"] = per_prompt_performance
    return results


def run_per_category_kmeans(dataset: str, embedding_type: str, base_dir: str):
    # load data
    dataset: Dataset = Dataset(
        base_dir=base_dir,
        dataset_name=dataset,
        embedding_type=embedding_type,
    )

    # compute concat per category performance
    # print("Computing per category performance")
    per_category_performance = {}
    for category in dataset.get_categories():
        dataset.set_category(category)

        embeddings = dataset.get_embeddings()
        labels = dataset.get_clustering_labels(category)
        n_clusters = len(set(labels))
        kmeans_result = kmeans(embeddings, labels, n_clusters=n_clusters)
        per_category_performance[category] = kmeans_result

    # save the results

    return per_category_performance


def run_nr_kmeans_clusterings(dataset: str, embedding_type: str, base_dir: str):

    # load data
    dataset: Dataset = Dataset(
        base_dir=base_dir,
        dataset_name=dataset,
        embedding_type=embedding_type,
    )

    # run nrkmeans
    nrkmeans_results = nrkmeans(
        data=dataset.get_embeddings(),
        labels=dataset.get_full_clustering_labels(),
        n_clusters=dataset.get_n_clusters_per_category(),
    )
    
    return nrkmeans_results

def run_enrc_clusterings(dataset: str, embedding_type: str, base_dir: str):

    # load data
    dataset: Dataset = Dataset(
        base_dir=base_dir,
        dataset_name=dataset,
        embedding_type=embedding_type,
    )

    save_dir = os.path.join(base_dir, "embedding_cache", embedding_type)

    # run nrkmeans
    nrkmeans_results = enrc(
        data=dataset.get_embeddings(),
        labels=dataset.get_full_clustering_labels(),
        n_clusters=dataset.get_n_clusters_per_category(),
        save_dir=save_dir
    )
    
    return nrkmeans_results

def run_orth1_clusterings(dataset: str, embedding_type: str, base_dir: str):

    # load data
    dataset: Dataset = Dataset(
        base_dir=base_dir,
        dataset_name=dataset,
        embedding_type=embedding_type,
    )

    # run nrkmeans
    orth_results = orth(
        data=dataset.get_embeddings(),
        labels=dataset.get_full_clustering_labels(),
        n_clusters=dataset.get_n_clusters_per_category(),
        orth_type="orth_1"
    )
    
    return orth_results

def run_orth2_clusterings(dataset: str, embedding_type: str, base_dir: str):

    # load data
    dataset: Dataset = Dataset(
        base_dir=base_dir,
        dataset_name=dataset,
        embedding_type=embedding_type,
    )

    # run nrkmeans
    orth_results = orth(
        data=dataset.get_embeddings(),
        labels=dataset.get_full_clustering_labels(),
        n_clusters=dataset.get_n_clusters_per_category(),
        orth_type="orth_2"
    )
    
    return orth_results

