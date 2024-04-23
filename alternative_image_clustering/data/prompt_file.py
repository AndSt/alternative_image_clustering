import json
from abc import abstractmethod


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
        entries_subset = _PromptFileJson(
            [
                {"labels": labels, "prompts": [prompts]}
                for labels, prompts in entries_subset
            ]
        )
        return entries_subset

    def slice(self, slicing):
        entries = self.extract_prompts_with_labels()[slicing]
        entries_subset = _PromptFileJson(
            [{"labels": labels, "prompts": [prompts]} for labels, prompts in entries]
        )
        return entries_subset

    def __add__(self, other):
        entries = (
            self.extract_prompts_with_labels() + other.extract_prompts_with_labels()
        )
        return _PromptFileJson(
            (
                {"prompts": [prompt]}
                if label is None
                else {"labels": label, "prompts": [prompt]}
            )
            for label, prompt in entries
        )


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
        return list(
            (_get_labels(entry["labels"]) if "labels" in entry else None, prompt)
            for entry in self
            for prompt in entry["prompts"]
        )

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
        return _PromptFileJson(
            combine_prompts([entries[p_id] for p_id in ids]) for ids in comb_ids
        )

    def filter_category(self, category):
        entries = self.extract_prompts_with_labels()
        subset_ids = [
            p_id
            for p_id, (labels, _) in enumerate(entries)
            if labels is not None and category in labels
        ]
        entries_subset = _PromptFileJson(
            [
                {
                    "labels": [category],
                    "prompts": [entries[p_id][1] for p_id in subset_ids],
                }
            ]
        )
        return subset_ids, entries_subset


def load_prompt_file(path) -> PromptFile:
    with open(path, "r") as filein:
        try:
            file = json.load(filein)
            return _PromptFileJson(file)
        except json.decoder.JSONDecodeError as e:
            print(e)
            print(
                "[Info] Prompts could not be loaded with json, continuing with the old mode."
            )
            prompts = filein.read().split("\n")
            if prompts[-1] == "":
                prompts.pop()
            return _PromptFileList(prompts)
