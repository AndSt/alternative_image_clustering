import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from typing import Counter
import ClusterEnsembles as CE

from alternative_image_clustering.data.dataset import Dataset
from alternative_image_clustering.clustering import run_single_kmeans
from alternative_image_clustering.metrics import get_multiple_labeling_metrics


def label_matrix_to_linkage_and_distance_clustering(
    cluster_preds, threshold: float = 0.5, similarity_metric="ami"
):
    if not isinstance(cluster_preds, np.ndarray):
        cluster_preds = np.array(cluster_preds)

    if similarity_metric == "ami":
        sim_fct = adjusted_mutual_info_score
    elif similarity_metric == "nmi":
        sim_fct = normalized_mutual_info_score

    similarities = np.ones((cluster_preds.shape[0], cluster_preds.shape[0]))
    for i in range(len(cluster_preds) - 1):
        for j in range(i, len(cluster_preds)):
            similarities[i, j] = similarities[j, i] = sim_fct(
                cluster_preds[i], cluster_preds[j]
            )

    dists = 1 - similarities
    np.fill_diagonal(dists, 0)
    Z = linkage(squareform(dists), "single", "euclidean")
    distance_clustering = fcluster(Z, threshold, "distance")

    return Z, distance_clustering


def distance_clustering_to_splitting(distance_clustering):
    clust_ids, counts = np.unique(distance_clustering, return_counts=True)

    splitting = []
    for cid, count in zip(clust_ids, counts):
        if count < 2:
            continue
        indexes = np.arange(distance_clustering.shape[0])[distance_clustering == cid]
        splitting.append(indexes)
    return splitting


def get_consensus_prediction(dataset, splitting, predictions, random_state: int = 42):
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)

    pred_labels = []

    for split in splitting:
        cat_labels = predictions[split]
        labels_ce = CE.cluster_ensembles(cat_labels, solver="mcla", random_state=random_state)
        pred_labels.append(labels_ce)
    return pred_labels


def get_selected_prompts_prediction(dataset, splitting, predictions, random_state: int=42):
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)

    pred_labels = []
    for split in splitting:
        dataset.set_indices(split.tolist())
        embeddings = dataset.get_embeddings()

        n_clusters = [len(set(predictions[i])) for i in range(len(predictions))]
        n_clusters = Counter(n_clusters).most_common(1)[0][0]

        kmeans_results = run_single_kmeans(data=embeddings, n_clusters=n_clusters, random_state=random_state)
        pred_labels.append(kmeans_results["labels"])

    return pred_labels


def run_pgacc(
    dataset_name: str,
    base_dir: str,
    embedding_type: str = "tfidf",
    aggregation_strategy: str = "consensus",
    threshhold: float = 0.4,
    similarity_metric="ami",
    random_state: int = 42
):

    if embedding_type not in ["tfidf", "sbert_concat"]:
        raise ValueError("Invalid embedding type.")

    # load data
    dataset = Dataset(
        base_dir=base_dir,
        dataset_name=dataset_name,
        embedding_type=embedding_type,
    )

    embeddings = dataset.get_embeddings()

    per_prompt_preds = []
    all_names = []

    print("Start initial")
    for category in dataset.get_categories():
        category_ids = dataset.get_category_ids(category)

        labels = dataset.get_clustering_labels(category)  # [0:400]
        n_clusters = len(set(labels))

        # img_kmeans_results = kmeans(image_embeddings, labels, n_clusters=n_clusters)
        # per_prompt_preds.append(img_kmeans_results["best_labels"])
        # all_names.append(f"prompt_{category}_image")

        for category_id in category_ids:
            dataset.set_indices(category_id)
            embeddings = dataset.get_embeddings()  # [0:400]
            kmeans_result = run_single_kmeans(data=embeddings, n_clusters=n_clusters, random_state=random_state)

            per_prompt_preds.append(kmeans_result["labels"])
            all_names.append({
                "category": category,
                "prompt_id": category_id
            })

    print("Start linkage")
    Z, distance_clustering = label_matrix_to_linkage_and_distance_clustering(
        cluster_preds=per_prompt_preds,
        threshold=threshhold,
        similarity_metric=similarity_metric,
    )
    splitting = distance_clustering_to_splitting(distance_clustering)

    print("Adaption")
    if aggregation_strategy == "consensus":
        pred_labels = get_consensus_prediction(
            dataset=dataset, splitting=splitting, predictions=per_prompt_preds, random_state=random_state)
    elif aggregation_strategy == "selection":
        pred_labels = get_selected_prompts_prediction(
            dataset, splitting, per_prompt_preds, random_state=random_state
        )
    else:
        raise ValueError("Invalid aggregation strategy.")

    results = {
        "aggregation_strategy": aggregation_strategy,
        "threshold": threshhold,
        "Z_matrix": Z,
        "similarity_metric": similarity_metric,
        "pred_labels": pred_labels,
        "metrics": get_multiple_labeling_metrics(
            labels_true=dataset.get_full_clustering_labels(),
            labels_pred=np.array(pred_labels).T,
            categories=dataset.get_categories(),
        ),
        "splitting": splitting,
        "names": all_names,
    }

    return results
