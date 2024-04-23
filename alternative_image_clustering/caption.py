import glob
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    InstructBlipForConditionalGeneration,
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
)

from alternative_image_clustering.data.dataset import load_prompt_file


def _get_img_paths(img_dir, sample_size=None):
    img_paths = np.array(
        glob.glob(img_dir + "**/*.jpg", recursive=True)
        + glob.glob(img_dir + "**/*.ppm", recursive=True)
        + glob.glob(img_dir + "**/*.png", recursive=True)
    )

    if sample_size is not None:
        sample_inds = np.arange(
            0, len(img_paths), len(img_paths) // sample_size, dtype=int
        )
        sample = img_paths[sample_inds]
        np.random.shuffle(sample)
        return sample

    np.random.shuffle(img_paths)

    return img_paths


def _save_results(results, file_path):
    with open(file_path, "w") as file:
        json.dump(results, file, indent=4)


class VQA:
    def __init__(self, model_name="Salesforce/instructblip-vicuna-7b"):
        self.device = torch.device("cuda")
        self.model_name = model_name
        self.save_name = self.model_name.split("/")[1]
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            self.model_name, device_map="auto"
        )
        self.model.to(self.device)

    def generate(self, dataset_name, prompt_version, sample_size=None):
        dataset_dir = f"datasets/{dataset_name}"
        img_dir = f"{dataset_dir}/images/"
        prompt_path = f"{dataset_dir}/prompts_{prompt_version}.txt"

        img_paths = _get_img_paths(img_dir, sample_size=sample_size)
        prompts = load_prompt_file(prompt_path).extract_prompts()

        results = {
            img_path.split("/images/")[-1]: self._generate_answers(img_path, prompts)
            for img_path in tqdm(img_paths, desc="Generating captions")
        }

        file_path = f"{dataset_dir}/results_{self.save_name}_{prompt_version}.txt"
        _save_results(results, file_path)

    def _generate_answers(self, image_path, questions):
        raw_image = Image.open(image_path)

        @torch.no_grad()
        def ask(prompt):
            inputs = self.processor(
                images=raw_image, text=prompt, return_tensors="pt"
            ).to(self.device)

            out = self.model.generate(
                **inputs,
                num_beams=5,
                max_length=120,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,
            )

            decoded_answer = self.processor.batch_decode(out, skip_special_tokens=True)
            return decoded_answer[0]

        answers = list(map(ask, questions))
        return answers


class CustomCollator:
    def __init__(self):
        pass

    def __call__(self, batch):
        collated_batch = {}
        key_list = batch[0].keys()
        for key in key_list:
            collated_batch[key] = [item[key] for item in batch]
        return collated_batch


class LLavaVQA:
    def __init__(self, model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf", batch_size: int=32):

        self.model_name = model_name
        self.save_name = self.model_name.split("/")[1]

        self.batch_size = batch_size

        self.processor = LlavaNextProcessor.from_pretrained(model_name)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name, device_map="auto"
        )
        self.device = self.model.device

    def generate(self, dataset_name, prompt_version, sample_size=None):
        dataset_dir = f"datasets/{dataset_name}"
        img_dir = f"{dataset_dir}/images/"
        prompt_path = f"{dataset_dir}/prompts_{prompt_version}.txt"

        img_paths = _get_img_paths(img_dir, sample_size=sample_size)
        prompts = load_prompt_file(prompt_path).extract_prompts()

        input_prompts = []
        for idx, image_path in enumerate(img_paths):
            for prompt_idx, prompt in enumerate(prompts):
                input_prompts.append({
                    "image_idx": image_path.split("/images/")[-1],
                    "prompt_idx": prompt_idx,
                    "prompt": f"[INST]<image>\n{prompt}[/INST]",
                    "image": image_path
                })

        loader = DataLoader(input_prompts, batch_size=self.batch_size, collate_fn=CustomCollator())

        generated_texts = {}

        i = 0
        for batch in tqdm(loader):
            images = [Image.open(image_path) for image_path in batch["image"]]
            inputs = self.processor(batch["prompt"], images, return_tensors="pt", padding=True).to(self.device)

            # autoregressively complete prompt
            output = self.model.generate(**inputs, max_new_tokens=200, pad_token_id=self.processor.tokenizer.eos_token_id)

            for idx, out in enumerate(output):
                generated_text = self.processor.decode(out, skip_special_tokens=True).split("[/INST]")[1]

                image_idx = batch["image_idx"][idx]
                prompt_idx = batch["prompt_idx"][idx]
                if image_idx not in generated_texts:
                    generated_texts[image_idx] = {}
                generated_texts[image_idx][prompt_idx] = generated_text

        # transform back to original format
        results = {}
        for image_idx, prompt_dict in generated_texts.items():
            results[image_idx] = []
            for i in range(len(prompt_dict)):
                results[image_idx].append(prompt_dict[i])

        file_path = f"{dataset_dir}/results_{self.save_name}_{prompt_version}.txt"
        _save_results(results, file_path)
