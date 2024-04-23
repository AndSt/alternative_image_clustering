import os
import json
from typing import Optional

import joblib
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from alternative_image_clustering.data.prompt_file import PromptFile


class GeneratedTextObj(dict):
    """Stores results stored as dictionary loaded from the results json file."""

    def extract_answers(self):
        return np.array(list(self.values()))

    def extract_image_paths(self):
        return np.array(list(self.keys()))

    def subset(self, ids):
        return GeneratedTextObj(
            {
                image_path: [results[p_id] for p_id in ids]
                for image_path, results in self.items()
            }
        )

    def combine(self, comb_ids):
        return GeneratedTextObj(
            {
                path: [" ".join(np.array(results)[ids]) for ids in comb_ids]
                for path, results in self.items()
            }
        )

    def check_overlap(self, other):
        self_images = self.extract_image_paths()
        other_images = other.extract_image_paths()

        if len(self_images) != len(other_images):
            return False

        for path in self_images:
            if path not in other_images:
                return False

        return True

    def __add__(self, other):
        if not self.check_overlap(other):
            raise Exception("The samples do not overlap.")

        return GeneratedTextObj(
            {path: results + other[path] for path, results in self.items()}
        )


def load_prompt_file(path):
    if not os.path.exists(path):
        return None

    prompt_dict = {}

    with open(path, "r") as filein:
        prompt_file = json.load(filein)
    i = 0
    for entry in prompt_file:
        assert len(entry["labels"]) == 1
        label = entry["labels"][0]
        for prompt in entry["prompts"]:
            prompt_dict[i] = {"prompt_id": i, "category": label, "prompt": prompt}
            i += 1
    return prompt_dict


DEBUG = False


class Dataset:
    def __init__(
        self,
        dataset_name: str,
        prompt_version: Optional[str] = "v1",
        embedding_type: str = "image",
        t2i_model: str = "llava-v1.6-mistral-7b-hf",
        base_dir: str = "",
        prompt_indices: Optional[list] = None,
        results: Optional[GeneratedTextObj] = None,
        prompt_file: Optional[PromptFile] = None,
    ):
        if prompt_version is None and (results is None or prompt_file is None):
            raise Exception(
                "prompt_version or results and prompt_file need(s) to be specified"
            )

        self.dataset_name = dataset_name
        self.prompt_version = prompt_version
        self.t2i_model = t2i_model
        self.base_dir = base_dir
        self.dataset_dir = os.path.join(base_dir, "datasets", dataset_name)
        self.label_dir = os.path.join(self.dataset_dir, "clustering")

        # self.results_path = f"{self.dataset_dir}/results_{prompt_version}.txt"
        self.results_path = os.path.join(
            self.dataset_dir, f"results_{self.t2i_model}_{prompt_version}.txt"
        )
        self.prompts_path = os.path.join(
            self.dataset_dir, f"prompts_{prompt_version}.txt"
        )

        self.prompt_dict = load_prompt_file(self.prompts_path)
        self.results = self._load_results()

        self.generated_text = self.results.extract_answers()
        self.image_paths = self.results.extract_image_paths()

        if "images/" not in self.image_paths[0]:
            self.img_names = [path.split("images/")[-1] for path in self.image_paths]
        else:
            self.img_names = self.image_paths

        if DEBUG:
            self.img_names = self.img_names[:400]

        with open(
            os.path.join(self.dataset_dir, "cluster_to_name.json"), "r"
        ) as filein:
            self.clusters_to_names = json.load(filein)

        if prompt_indices is None:
            self.active_prompt_indices = list(self.prompt_dict.keys())
        else:
            self.active_prompt_indices = prompt_indices

        self.embedding_type = embedding_type

    def _load_results(self) -> GeneratedTextObj:
        with open(self.results_path, "r") as filein:
            json_obj = json.load(filein)

        if isinstance(json_obj, list):
            return GeneratedTextObj(
                {path: results for entry in json_obj for path, results in entry.items()}
            )

        return GeneratedTextObj(json_obj)

    def get_n_clusters(self):
        n_clusters = {
            category: len(self.clusters_to_names[category].keys())
            for category in self.get_categories()
        }
        return n_clusters

    def get_category_ids(self, category):
        subset_ids = []
        for _, entry in self.prompt_dict.items():
            if entry["category"] == category:
                subset_ids.append(entry["prompt_id"])
        return subset_ids

    def set_category(self, category):
        self.active_prompt_indices = self.get_category_ids(category)

    def get_categories(self):
        return list(self.clusters_to_names.keys())

    def get_clustering_labels(self, category):
        with open(f"{self.label_dir}/clusters_{category}.json", "r") as filein:
            clustering = json.load(filein)

        return [clustering[img_name] for img_name in self.img_names]

    def get_embeddings(self, sbert_model=None):
        # return np.load(self.cache_file_name)

        # embedding cache needs: dataset_name, embedding_type, prompt_version

        cache_dir = os.path.join(self.base_dir, "embedding_cache", self.embedding_type)
        os.makedirs(cache_dir, exist_ok=True)

        # check images first
        if self.embedding_type == "image":
            image_file_name = os.path.join(cache_dir, f"{self.dataset_name}_image.pbz2")
            embedding_file = joblib.load(image_file_name)
            embeddings = np.array(
                [embedding_file[img_name] for img_name in self.img_names]
            )

            if DEBUG:
                return embeddings[:400]
            return embeddings

        # check tfidf
        active_prompts = "_".join([str(e) for e in self.active_prompt_indices])
        file_name = f"{self.dataset_name}_{self.prompt_version}_{active_prompts}.pbz2"
        file_name = os.path.join(cache_dir, file_name)

        if os.path.exists(file_name):
            return joblib.load(file_name)

        if self.embedding_type == "tfidf":  # always concatenated text
            texts = []
            for img_name in self.img_names:
                active_texts = [
                    self.results[img_name][i].strip()
                    for i in self.active_prompt_indices
                ]
                texts.append(" ".join(active_texts))

            vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
            embeddings = vectorizer.fit_transform(texts)
            joblib.dump(embeddings, file_name)

            if DEBUG:
                return embeddings[:400]

            return embeddings
        # load textual concatenation of a subset of the prompts
        elif self.embedding_type == "sbert_concat":
            texts = []
            for img_name in self.img_names:
                active_texts = [
                    self.results[img_name][i].strip()
                    for i in self.active_prompt_indices
                ]
                texts.append(" ".join(active_texts))
            if sbert_model is None:
                model = SentenceTransformer("thenlper/gte-large", device="cuda")
            else:
                model = sbert_model
            embeddings = model.encode(texts, show_progress_bar=True)

            joblib.dump(embeddings, file_name)
            if DEBUG:
                return embeddings[:400]
            return embeddings

        # load average of the embeddings of a subset of the prompts
        elif self.embedding_type == "sbert_avg":
            file_name = f"{self.dataset_name}_{self.prompt_version}.pbz2"
            file_name = os.path.join(cache_dir, file_name)
            if os.path.exists(file_name):
                embeddings = joblib.load(file_name)
                assert len(self.prompt_dict) == len(embeddings)
                for prompt in self.prompt_dict:
                    assert embeddings[prompt].shape == (len(self.img_names), 1024)
            else:
                texts = {}
                for prompt in self.prompt_dict:
                    texts[prompt] = []
                    for img_name in self.img_names:
                        texts[prompt].append(self.results[img_name][prompt].strip())

                embeddings = {}
                model = SentenceTransformer("thenlper/gte-large", device="cuda")
                for prompt in self.active_prompt_indices:
                    embs = model.encode(texts[prompt], show_progress_bar=True)
                    embeddings[prompt] = embs
                joblib.dump(embeddings, file_name)

            embeddings = np.mean(
                [embeddings[i] for i in self.active_prompt_indices], axis=0
            )

            if DEBUG:
                return embeddings[:400]
            return embeddings
