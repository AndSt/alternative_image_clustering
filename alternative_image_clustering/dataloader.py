import json
from abc import ABC, abstractmethod
from itertools import combinations
from typing import Optional

import numpy as np


def _get_labels(label):
    if isinstance(label, str):
        return [label]
    return label
 

class PromptFile(list):
    @abstractmethod
    def extract_prompts(self) -> list:
        pass

    @abstractmethod
    def extract_prompts_with_labels(self) -> list:
        pass

    @abstractmethod
    def combine(self, comb_ids):
        pass

    @abstractmethod
    def filter_category(self, category):
        pass

    def subset(self, indexes):
        entries = self.extract_prompts_with_labels()
        entries_subset = [entries[ind] for ind in indexes]
        entries_subset = _PromptFileJson([{"labels": labels, "prompts": [prompts]} for labels, prompts in entries_subset])
        return entries_subset

    def slice(self, slicing):
        entries = self.extract_prompts_with_labels()[slicing]
        entries_subset = _PromptFileJson([{"labels": labels, "prompts": [prompts]} for labels, prompts in entries])
        return entries_subset

    def __add__(self, other):
        entries = self.extract_prompts_with_labels() + other.extract_prompts_with_labels()
        return _PromptFileJson(({"prompts": [prompt]} if label is None else {"labels": label, "prompts": [prompt]})
                               for label, prompt in entries)


class _PromptFileList(PromptFile):
    def extract_prompts(self):
        return self

    def extract_prompts_with_labels(self):
        # None: the old file format does not have label category data
        return [(None, prompt) for prompt in self]

    def combine(self, comb_ids):
        return _PromptFileList(" ".join(self[p_id] for p_id in ids) for ids in comb_ids)

    def filter_category(self, category):
        # the old file format does not have label category data
        return _PromptFileList()


class _PromptFileJson(PromptFile):
    def extract_prompts(self):
        return list(prompt for entry in self for prompt in entry["prompts"])

    def extract_prompts_with_labels(self):
        return list((_get_labels(entry["labels"]) if "labels" in entry else None, prompt)
                    for entry in self for prompt in entry["prompts"])

    def combine(self, comb_ids):
        def combine_prompts(prompts):
            comb_labels = set()
            comb_prompt = ""

            for labels, prompt in prompts:
                if labels is not None:
                    comb_labels.update(set(_get_labels(labels)))
                comb_prompt += " " + prompt

            comb_entry = {}
            if len(comb_labels) > 0:
                comb_entry["labels"] = list(comb_labels)
            comb_entry["prompts"] = [comb_prompt]
            return comb_entry

        entries = self.extract_prompts_with_labels()
        return _PromptFileJson(combine_prompts([entries[p_id] for p_id in ids]) for ids in comb_ids)

    def filter_category(self, category):
        entries = self.extract_prompts_with_labels()
        subset_ids = [p_id for p_id, (labels, _) in enumerate(entries) if labels is not None and category in labels]
        entries_subset = _PromptFileJson([{"labels": [category], "prompts": [entries[p_id][1] for p_id in subset_ids]}])
        return subset_ids, entries_subset


def load_prompt_file(path) -> PromptFile:
    with open(path, "r") as filein:
        try:
            file = json.load(filein)
            return _PromptFileJson(file)
        except json.decoder.JSONDecodeError as e:
            print(e)
            print("[Info] Prompts could not be loaded with json, continuing with the old mode.")
            prompts = filein.read().split("\n")
            if prompts[-1] == '':
                prompts.pop()
            return _PromptFileList(prompts)


class Result(dict):
    """Stores results stored as dictionary loaded from the results json file."""

    def extract_answers(self):
        return np.array(list(self.values()))

    def extract_image_paths(self):
        return np.array(list(self.keys()))

    def subset(self, ids):
        return Result({image_path: [results[p_id] for p_id in ids] for image_path, results in self.items()})

    def combine(self, comb_ids):
        return Result({path: [" ".join(np.array(results)[ids]) for ids in comb_ids]
                       for path, results in self.items()})

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

        return Result({path: results + other[path] for path, results in self.items()})


class Dataset:
    def __init__(self, dataset_name: str, prompt_version: Optional[str], results: Optional[Result],
                 prompt_file: Optional[PromptFile]):
        if prompt_version is None and (results is None or prompt_file is None):
            raise Exception("prompt_version or results and prompt_file need(s) to be specified")

        self.dataset_name = dataset_name
        self.prompt_version = prompt_version
        self.dataset_dir = f"datasets/{dataset_name}"
        self.label_dir = f"{self.dataset_dir}/clustering"

        if prompt_version is None:
            self.results_path = None
            self.prompts_path = None
        else:
            self.results_path = f"{self.dataset_dir}/results_{prompt_version}.txt"
            self.prompts_path = f"{self.dataset_dir}/prompts_{prompt_version}.txt"

        if results is None:
            self.results = self._load_results()
            self.prompt_file = load_prompt_file(self.prompts_path)
        else:
            self.results = results
            self.prompt_file = prompt_file

        self.answers = self.results.extract_answers()
        self.image_paths = self.results.extract_image_paths()

    @classmethod
    def load_from_disk(cls, dataset_name, prompt_version):
        return cls(dataset_name, prompt_version, None, None)

    @classmethod
    def _load_custom(cls, dataset_name, results, prompt_file):
        return cls(dataset_name, None, results, prompt_file)

    def _load_results(self) -> Result:
        with open(self.results_path, "r") as filein:
            json_obj = json.load(filein)

        if isinstance(json_obj, list):
            return Result({path: results for entry in json_obj for path, results in entry.items()})

        return Result(json_obj)

    def get_clustering(self, category):
        with open(f"{self.label_dir}/clusters_{category}.json", "r") as filein:
            clustering = json.load(filein)
        img_names = self.image_paths
        if "images/" in self.image_paths[0]:
            img_names = [path.split("images/")[-1] for path in self.image_paths]
        return [clustering[img_name] for img_name in img_names]

    def get_n_clusters(self):
        n_clusters = []
        cache = {}
        for labels, _ in self.prompt_file.extract_prompts_with_labels():
            if labels is None:
                n_clusters.append([-1])
                continue

            n_clusters_ = []
            for label in labels:
                if label not in cache:
                    with open(f"{self.label_dir}/clusters_{label}.json", "r") as filein:
                        clusters_file: dict = json.load(filein)
                        uniques, counts = np.unique(np.array(list(clusters_file.values())), return_counts=True)
                        cache[label] = sum(counts >= np.sum(counts) / 10)

                n_clusters_.append(cache[label])
            n_clusters.append(n_clusters_)
        return n_clusters
        
    def slice(self, slicing: slice):
        subset_ids = np.arange(self.answers.shape[1])[slicing]
        prompt_file_subset = self.prompt_file.slice(slicing)
        results_subset = self.results.subset(subset_ids)
        return self._load_custom(self.dataset_name, results_subset, prompt_file_subset)

    def select(self, indexes):
        subset_ids = np.array(indexes)
        prompt_file_subset = self.prompt_file.subset(subset_ids)
        results_subset = self.results.subset(subset_ids)
        return self._load_custom(self.dataset_name, results_subset, prompt_file_subset)
        
    def filter_category(self, category):
        subset_ids, prompt_file_subset = self.prompt_file.filter_category(category)
        results_subset = self.results.subset(subset_ids)
        return self._load_custom(self.dataset_name, results_subset, prompt_file_subset)

    def combine(self, max_level=None):
        num_samples, num_answers = self.answers.shape[0], self.answers.shape[1]
        if max_level is None:
            max_level = num_answers

        comb_ids = [list(comb) for i in range(1, max_level + 1) for comb in combinations(range(max_level), i)]
        results_new = self.results.combine(comb_ids)
        prompt_file_new = self.prompt_file.combine(comb_ids)

        dataset_new = self._load_custom(self.dataset_name, results_new, prompt_file_new)
        return dataset_new

    def __add__(self, other):
        comb_results = self.results + other.results
        comb_prompt_file = self.prompt_file + other.prompt_file
        return self._load_custom(self.dataset_name, comb_results, comb_prompt_file)

    def save_as(self, prompt_version):
        prompts_path = f"{self.dataset_dir}/prompts_{prompt_version}.txt"
        with open(prompts_path, "w") as file_out:
            json.dump(self.prompt_file, file_out, indent=4)

        results_path = f"{self.dataset_dir}/results_{prompt_version}.txt"
        with open(results_path, "w") as file_out:
            json.dump(self.results, file_out, indent=4)

        return self.load_from_disk(self.dataset_name, prompt_version)

