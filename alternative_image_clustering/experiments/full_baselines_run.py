import argparse
import os
import joblib
from tqdm import tqdm

from alternative_image_clustering.experiments.single_experiments import (
    run_per_prompt_kmeans,
    run_per_category_kmeans,
    run_full_kmeans,
    run_nr_kmeans_clusterings,
)



def main(base_dir: str = "data", dataset: str = None):
    cache_dir = "benchmark_cache_stddev"
    datasets = ["cards", "fruit360", "nrobjects", "gtsrb"]
    if dataset in datasets:
        datasets = [dataset]

    num_exps = len(datasets) * (2 + 2 * 4)
    pbar = tqdm(total=num_exps)

    for dataset in datasets:
        save_dir = os.path.join(base_dir, cache_dir, dataset, "image")
        os.makedirs(save_dir, exist_ok=True)

        print(f"Running experiments for dataset {dataset}")
        image_file = os.path.join(save_dir, "kmeans.pbz2")
        if not os.path.exists(image_file):
            results = run_full_kmeans(dataset, embedding_type="image", base_dir=base_dir)
            joblib.dump(results, image_file)
        pbar.update(1)

        nrkmeans_file = os.path.join(save_dir, "nrkmeans.pbz2")
        if not os.path.exists(nrkmeans_file):
            results = run_nr_kmeans_clusterings(
                dataset, embedding_type="image", base_dir=base_dir
            )
            joblib.dump(results, nrkmeans_file)
        pbar.update(1)

        for embedding_type in ["tfidf", "sbert_concat"]:
            save_dir = os.path.join(
                base_dir, cache_dir, dataset, embedding_type
            )
            os.makedirs(save_dir, exist_ok=True)

            full_kmeans_file = os.path.join(save_dir, "full_kmeans.pbz2")
            if not os.path.exists(full_kmeans_file):
                results = run_full_kmeans(dataset, embedding_type, base_dir=base_dir)
                joblib.dump(results, full_kmeans_file)
            pbar.update(1)

            per_prompt_file = os.path.join(save_dir, "per_prompt_kmeans.pbz2")
            if not os.path.exists(per_prompt_file):
                results = run_per_prompt_kmeans(
                    dataset, embedding_type, base_dir=base_dir
                )
                joblib.dump(results, per_prompt_file)
            pbar.update(1)

            per_category_file = os.path.join(save_dir, "per_category_kmeans.pbz2")
            if not os.path.exists(per_category_file):
                results = run_per_category_kmeans(
                    dataset, embedding_type, base_dir=base_dir
                )
                joblib.dump(results, per_category_file)
            pbar.update(1)

            nrkmeans_file = os.path.join(save_dir, "nrkmeans.pbz2")
            if not os.path.exists(nrkmeans_file):
                results = run_nr_kmeans_clusterings(
                    dataset, embedding_type, base_dir=base_dir
                )
                joblib.dump(results, nrkmeans_file)
            pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alternative Image Clustering")
    parser.add_argument(
        "--base_dir", type=str, default="/mnt/data/stephana93dm/storage/projects/alternative_image_clustering", help="Base directory for data"
    )
    parser.add_argument(
        "--dataset", type=str, default=None, help="Dataset to run experiments on"
    )

    args = parser.parse_args()

    main(args.base_dir, dataset=args.dataset)
