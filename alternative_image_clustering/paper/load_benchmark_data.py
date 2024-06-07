import os
from sre_parse import CATEGORIES
import joblib

from typing import List
from alternative_image_clustering.data.dataset import Dataset


def create_kmeans_row(
    base_dir: str,
    embedding_type: str,
    datasets: List[str],
    metrics: List[str] = ["ACC", "AMI"],
    file_name: str = "kmeans",
):
    row = [
        embedding_type.replace("sbert_concat", "SBERT").replace("tfidf", "TF-IDF"),
        file_name.replace("_", "-").replace("per-category-kmeans", "Concatenate"),
    ]
    for dataset in datasets:
        dset: Dataset = Dataset(
            dataset_name=dataset, base_dir=base_dir, embedding_type=embedding_type
        )
        categories = dset.get_categories()
        if embedding_type == "image":
            load_dir = "benchmark_cache_old"
        else:
            load_dir = "benchmark_cache_old"
        results = joblib.load(
            os.path.join(
                base_dir,
                load_dir, #  here old???
                dataset,
                embedding_type,
                f"{file_name}.pbz2",
            )
        )
        for category in categories:
            for metric in metrics:
                row.append(results[category]["metrics"][metric])

    return row


def create_nr_baseline_row(
    base_dir: str,
    embedding_type: str,
    baseline: str,
    datasets: List[str],
    metrics=["ACC", "AMI"],
):
    row = [embedding_type.replace("sbert_concat", "SBERT").replace("tfidf", "TF-IDF"), baseline.replace("_", "-").replace("nrkmeans", "NR-kmeans")]
    for dataset in datasets:
        dset: Dataset = Dataset(
            dataset_name=dataset, base_dir=base_dir, embedding_type=embedding_type
        )
        categories = dset.get_categories()

        results = joblib.load(
            os.path.join(
                base_dir,
                "benchmark_cache_old",
                dataset,
                embedding_type,
                f"{baseline}.pbz2",
            )
        )
        for category in categories:
            for metric in metrics:
                row.append(results["metrics"][category][metric])

    return row


def create_per_prompt_rows(
    base_dir: str, embedding_type: str, datasets: List[str], metrics=["ACC", "AMI"]
):
    row_avg = [embedding_type.replace("sbert_concat", "SBERT").replace("tfidf", "TF-IDF"), "Avg. Prompt"]
    row_max = [embedding_type.replace("sbert_concat", "SBERT").replace("tfidf", "TF-IDF"), "Max. Prompt"]
    for dataset in datasets:
        dset: Dataset = Dataset(
            dataset_name=dataset, base_dir=base_dir, embedding_type=embedding_type
        )
        categories = dset.get_categories()

        results = joblib.load(
            os.path.join(
                base_dir,
                "benchmark_cache_old",
                dataset,
                embedding_type,
                "per_prompt_kmeans.pbz2",
            )
        )
        # print(results)
        for category in categories:
            for metric in metrics:
                row_avg.append(
                    results["per_category_avg_performance"][category][metric]
                )
                row_max.append(
                    results["per_category_max_performance"][category][metric]
                )

    return [row_avg]#, row_max]


def create_table_head(base_dir: str, datasets: List[str], metrics: List[str]):

    header = [("", "", ""), ("", "", "")]
    for dataset in datasets:
        dset: Dataset = Dataset(
            dataset_name=dataset, base_dir=base_dir, embedding_type="image"
        )
        categories = dset.get_categories()
        for category in categories:
            for metric in metrics:
                header.append((dataset, category, metric))

    return header
