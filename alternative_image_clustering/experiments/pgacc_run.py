import os
import argparse

import random
import joblib
import numpy as np

from alternative_image_clustering.data.dataset import Dataset
from alternative_image_clustering.pgacc import run_pgacc

RANDOM_STATE = 812


def run_pgacc(
    dataset_name: str,
    base_dir: str,
    embedding_type: str = "tfidf",
    aggregation_strategy: str = "consensus",
    threshhold: float = 0.4,
    similarity_metric="ami",
    num_runs: int = 10,
):

    random.seed(RANDOM_STATE)
    random_states = random.sample(range(1000), num_runs)

    runs = []
    for random_state in random_states:
        results = run_pgacc(
            base_dir=base_dir,
            dataset_name=dataset_name,
            embedding_type=embedding_type,
            aggregation_strategy=aggregation_strategy,
            threshhold=threshhold,
            similarity_metric=similarity_metric,
            random_state=random_state,
        )
        runs.append(results)

    dataset = Dataset(
        base_dir=base_dir, dataset_name=dataset_name, embedding_type=embedding_type
    )
    metrics = {category: {} for category in dataset.get_categories()}

    for run in runs:
        for category in dataset.get_categories():
            for metric in run[category]:
                if metric not in metrics[category]:
                    metrics[category][metric] = []
                metrics[category][metric].append(run[category][metric])

    avg_metrics = {
        category: {metric: np.mean(values) for metric, values in metrics[category].items()}
    }
    std_metrics = {
        category: {metric: np.std(values) for metric, values in metrics[category].items()}
    }
    result = runs[0]
    result["metrics"] = avg_metrics
    result["metrics_stddev"] = std_metrics

    return result


def main(base_dir: str, dataset: str):
    cache_dir = "benchmark_cache_stddev"
    datasets = ["cards", "fruit360", "nrobjects", "gtsrb"]
    if dataset in datasets:
        datasets = [dataset]

    embedding_types = ["image", "tfidf", "sbert"]
    aggregation_strategies = ["consensus", "selection"]
    similarity_metrics = ["ami", "nmi"]
    threshholds = [0.4]

    for dataset in datasets:
        for embedding_type in embedding_types:

            save_dir = os.path.join(base_dir, cache_dir, dataset, embedding_type)
            os.makedirs(save_dir, exist_ok=True)

            for aggregation_strategy in aggregation_strategies:
                for similarity_metric in similarity_metrics:
                    for threshhold in threshholds:
                        save_threshhold = f"{threshhold}".replace(".", "")
                        save_file = os.path.join(
                            save_dir,
                            f"pgacc_{aggregation_strategy}_{similarity_metric}_{save_threshhold}.pbz2",
                        )

                        if not os.path.exists(save_file):
                            results = run_pgacc(
                                dataset_name=dataset,
                                base_dir=base_dir,
                                embedding_type=embedding_type,
                                aggregation_strategy=aggregation_strategy,
                                threshhold=threshhold,
                                similarity_metric=similarity_metric,
                            )
                            joblib.dump(results, save_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alternative Image Clustering")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/mnt/data/stephana93dm/storage/projects/alternative_image_clustering",
        help="Base directory for data",
    )
    parser.add_argument(
        "--dataset", type=str, default=None, help="Dataset to run experiments on"
    )

    args = parser.parse_args()

    main(args.base_dir, dataset=args.dataset)
