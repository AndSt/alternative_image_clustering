import argparse
import os
import json
from tqdm import tqdm

from alternative_image_clustering.experiments.single_experiments import (
    run_image_experiments,
    run_per_prompt_kmeans,
    run_per_category_kmeans,
    run_full_kmeans,
    run_nr_kmeans_clusterings,
)


def save_json(data, path):
    with open(path, "w") as fileout:
        json.dump(data, fileout, indent=4)


def main(base_dir: str = "data"):

    datasets = ["cards", "fruit360", "nrbojects", "gtsrb"]

    num_exps = len(datasets) * (1 + 2 * 4)
    pbar = tqdm(total=num_exps)

    for dataset in datasets:
        save_dir = os.path.join(base_dir, "benchmark_cache", dataset)
        os.makedirs(save_dir, exist_ok=True)

        print(f"Running experiments for dataset {dataset}")
        save_dir = os.path.join(base_dir, "benchmark_cache", dataset, "image")
        image_file = os.path.join(save_dir, "image_kmeans.json")
        if not os.path.exists(image_file):
            results = run_image_experiments(dataset, base_dir="data")
            save_json(results, image_file)
        pbar.update(1)

        for embedding_type in ["tfidf", "sbert_concat"]:
            save_dir = os.path.join(
                base_dir, "benchmark_cache", dataset, embedding_type
            )
            os.makedirs(save_dir, exist_ok=True)

            full_kmeans_file = os.path.join(save_dir, "full_kmeans.json")
            if not os.path.exists(full_kmeans_file):
                results = run_full_kmeans(dataset, embedding_type, base_dir="data")
                save_json(results, full_kmeans_file)
            pbar.update(1)

            per_prompt_file = os.path.join(save_dir, "per_prompt_kmeans.json")
            if not os.path.exists(per_prompt_file):
                results = run_per_prompt_kmeans(
                    dataset, embedding_type, base_dir="data"
                )
                save_json(results, per_prompt_file)
            pbar.update(1)

            per_category_file = os.path.join(save_dir, "per_category_kmeans.json")
            if not os.path.exists(per_category_file):
                results = run_per_category_kmeans(
                    dataset, embedding_type, base_dir="data"
                )
                save_json(results, per_category_file)
            pbar.update(1)

            nrkmeans_file = os.path.join(save_dir, "nrkmeans.json")
            if not os.path.exists(nrkmeans_file):
                results = run_nr_kmeans_clusterings(
                    dataset, embedding_type, base_dir="data"
                )
                save_json(results, nrkmeans_file)
            pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alternative Image Clustering")
    parser.add_argument(
        "--base_dir", type=str, default="data", help="Base directory for data"
    )

    args = parser.parse_args()

    main(args.base_dir)
