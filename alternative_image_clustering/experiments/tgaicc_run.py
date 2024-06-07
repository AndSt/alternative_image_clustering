import os
import argparse
from tqdm import tqdm

import random
import joblib
import numpy as np

from alternative_image_clustering.data.dataset import Dataset
from alternative_image_clustering.tgaicc import run_tgaicc

RANDOM_STATE = 55


def run_single_tgaicc_experiment(
    dataset_name: str,
    base_dir: str,
    embedding_type: str = "tfidf",
    aggregation_strategy: str = "consensus",
    threshhold_strategy: str = "min",
    similarity_metric="ami",
    num_runs: int = 10,
):

    random.seed(RANDOM_STATE)
    random_states = random.sample(range(1000), num_runs)

    runs = []
    for random_state in random_states:
        try:
            results = run_tgaicc(
                base_dir=base_dir,
                dataset_name=dataset_name,
                embedding_type=embedding_type,
                aggregation_strategy=aggregation_strategy,
                threshhold_strategy=threshhold_strategy,
                similarity_metric=similarity_metric,
                random_state=random_state,
            )
            runs.append(results)
        except Exception as e:
            print(
                f"Error in {dataset_name}, {embedding_type}, {aggregation_strategy}, {threshhold_strategy}, {similarity_metric}"
            )
            print(e)

    if len(runs) < 4:
        raise ValueError("Not enough runs")

    dataset = Dataset(
        base_dir=base_dir, dataset_name=dataset_name, embedding_type=embedding_type
    )
    metrics = {category: {} for category in dataset.get_categories()}

    for run in runs:
        for category in dataset.get_categories():
            for metric in run["metrics"][category]:
                if metric not in metrics[category]:
                    metrics[category][metric] = []
                metrics[category][metric].append(run["metrics"][category][metric])

    avg_metrics = {
        category: {
            metric: np.mean(values) for metric, values in metrics[category].items()
        }
        for category in dataset.get_categories()
    }
    std_metrics = {
        category: {
            metric: np.std(values) for metric, values in metrics[category].items()
        }
        for category in dataset.get_categories()
    }

    result = runs[0]
    result["metrics"] = avg_metrics
    result["metrics_stddev"] = std_metrics

    return result


def main(base_dir: str, dataset: str):
    cache_dir = "benchmark_cache_pgaic"
    datasets = ["cards", "fruit360", "nrobjects", "gtsrb"]
    if dataset in datasets:
        datasets = [dataset]

    embedding_types = ["sbert_concat", "tfidf", "both"]
    aggregation_strategies = ["selection", "consensus"]
    similarity_metrics = ["ami", "nmi"]
    threshhold_strategies = ["min", "max"]

    num_exps = (
        len(datasets)
        * len(embedding_types)
        * len(aggregation_strategies)
        * len(similarity_metrics)
        * len(threshhold_strategies)
    )
    pbar = tqdm(total=num_exps)

    for dataset in datasets:
        for embedding_type in embedding_types:

            save_dir = os.path.join(base_dir, cache_dir, dataset, embedding_type)
            os.makedirs(save_dir, exist_ok=True)

            for aggregation_strategy in aggregation_strategies:
                for similarity_metric in similarity_metrics:
                    for threshhold_strategy in threshhold_strategies:
                        save_file = os.path.join(
                            save_dir,
                            f"tgaicc_{aggregation_strategy}_{similarity_metric}_{threshhold_strategy}.pbz2",
                        )

                        if not os.path.exists(save_file):
                            try:
                                results = run_single_tgaicc_experiment(
                                    dataset_name=dataset,
                                    base_dir=base_dir,
                                    embedding_type=embedding_type,
                                    aggregation_strategy=aggregation_strategy,
                                    threshhold_strategy=threshhold_strategy,
                                    similarity_metric=similarity_metric,
                                )
                                print(results)
                                joblib.dump(results, save_file)
                            except Exception as e:
                                print(
                                    f"Error in {dataset}, {embedding_type}, {aggregation_strategy}, {similarity_metric}, {threshhold_strategy}"
                                )
                                print(e)

                        pbar.update(1)


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
