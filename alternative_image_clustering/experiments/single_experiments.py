from audioop import avg
from nis import cat
from re import A
from IPython import embed
from click import prompt
from regex import D
from sympy import per
from alternative_image_clustering.data.dataset import Dataset

from alternative_image_clustering.clustering import kmeans, nrkmeans


def run_full_kmeans(dataset: str, embedding_type: str, base_dir: str):

    # load data
    dataset: Dataset = Dataset(
        base_dir=base_dir,
        dataset_name=dataset,
        embedding_type=embedding_type,
    )

    print("Running kmeans on full embeddings for each category")
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
    print("Running kmeans for each prompt")
    per_prompt_performance = {}
    for idx, prompt_info in dataset.prompt_dict.items():
        dataset.active_prompt_indices = [idx]
        embeddings = dataset.get_embeddings()
        labels = dataset.get_clustering_labels(prompt_info["category"])
        print(embeddings.shape, len(labels))
        kmeans_result = kmeans(embeddings, labels, n_clusters=len(set(labels)))
        per_prompt_performance[idx] = kmeans_result

    # compute per category performance
    print("Transforming to average and max. performance")
    per_category_avg_performance, per_category_max_performance = {}, {}

    for idx, info in per_prompt_performance.items():

        category = dataset.prompt_dict[idx]["category"]
        if category not in per_category_avg_performance:
            per_category_avg_performance[category] = {}
        if category not in per_category_max_performance:
            per_category_max_performance[category] = {}

        for metric in info["metrics"]:

            # initialize metric
            if metric not in per_category_avg_performance[category]:
                per_category_avg_performance[category][metric] = []
            if metric not in per_category_max_performance[category]:
                per_category_max_performance[category][metric] = []

            per_category_avg_performance[category][metric].append(
                info["metrics"][metric]
            )
            per_category_max_performance[category][metric].append(
                info["metrics"][metric]
            )

    for category in per_category_avg_performance:
        for metric in per_category_avg_performance[category]:
            per_category_avg_performance[category][metric] = sum(
                per_category_avg_performance[category][metric]
            ) / len(per_category_avg_performance[category][metric])
            per_category_max_performance[category][metric] = max(
                per_category_max_performance[category][metric]
            )

    return {
        # "full_kmeans_performance": full_kmeans_performance,
        # "per_prompt_performance": per_prompt_performance,
        "per_category_avg_performance": per_category_avg_performance,
        "per_category_max_performance": per_category_max_performance,
    }


def run_per_category_kmeans(dataset: str, embedding_type: str, base_dir: str):
    # load data
    dataset: Dataset = Dataset(
        base_dir=base_dir,
        dataset_name=dataset,
        embedding_type=embedding_type,
    )

    # compute concat per category performance
    print("Computing per category performance")
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
    results = {"nrkmeans": {}}

    nrkmeans_results = nrkmeans(
        data=dataset.get_embeddings(),
        labels=dataset.get_clustering_labels(),
        n_clusters=dataset.get_n_clusters(),
    )
    results["nrkmeans"] = nrkmeans_results
    
    return results


def run_image_experiments(dataset: str, base_dir: str):

    # load data
    dataset = Dataset(
        base_dir=base_dir,
        dataset_name=dataset,
        embedding_type="image",
    )

    embeddings = dataset.get_embeddings()

    # run kmeans for both all types of categories
    results = {"kmeans": {}, "nrkmeans": {}}
    for category in dataset.get_categories():
        labels = dataset.get_clustering_labels(category)
        n_clusters = len(set(labels))
        kmeans_result = kmeans(embeddings, labels, n_clusters=n_clusters)
        results["kmeans"][category] = kmeans_result

    # run nrkmeans
    # TODO: check input to nrkmeans
    nrkmeans_results = nrkmeans(
        data=embeddings,
        labels=dataset.get_clustering_labels(),
        n_clusters=dataset.get_n_clusters(),
    )
    results["nrkmeans"] = nrkmeans_results

    # run enrc

    # run orth / msc

    # save everything

    return results
