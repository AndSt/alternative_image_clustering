import glob
import json
import joblib
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


class CustomCollator:
    def __init__(self):
        pass

    def __call__(self, batch):
        collated_batch = {}
        key_list = batch[0].keys()
        for key in key_list:
            collated_batch[key] = [item[key] for item in batch]
        return collated_batch

def get_image_features(image, model, processor):

    inputs = processor(text=[""], images=[image], return_tensors="pt").to("cuda:0")
    pixel_values = inputs.pixel_values

    batch_size, num_patches, num_channels, height, width = pixel_values.shape
    reshaped_pixel_values = pixel_values.view(batch_size * num_patches, num_channels, height, width)
    image_features = model.vision_tower(reshaped_pixel_values)["pooler_output"].to("cpu")
    image_features = image_features.view(batch_size, num_patches, -1).mean(dim=1).detach().numpy()
    return image_features

class LLavaImageEmbedder:
    def __init__(
        self,
        model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        batch_size: int = 32,
    ):

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

        input_prompts = []
        for idx, image_path in enumerate(img_paths):
            input_prompts.append(
                {
                    "image_idx": image_path.split("/images/")[-1],
                    "prompt": "",
                    "image": image_path,
                }
            )

        loader = DataLoader(
            input_prompts, batch_size=self.batch_size, collate_fn=CustomCollator()
        )

        results = {}

        i = 0
        for batch in tqdm(loader):
            images = [Image.open(image_path) for image_path in batch["image"]]
            inputs = self.processor(
                batch["prompt"], images, return_tensors="pt", padding=True
            ).to(self.device)

            inputs = self.processor(text=batch["prompt"], images=images, return_tensors="pt")
            pixel_values = inputs.pixel_values

            batch_size, num_patches, num_channels, height, width = pixel_values.shape
            reshaped_pixel_values = pixel_values.view(batch_size * num_patches, num_channels, height, width)
            image_features = self.model.vision_tower(reshaped_pixel_values)["pooler_output"].to("cpu")
            image_features = image_features.view(batch_size, num_patches, -1).mean(dim=1).detach().numpy()

            for j in range(len(batch["image"])):
                image_idx = batch["image_idx"][j]
                assert image_idx not in results, "Image index already in results."
                results[image_idx] = image_features[j]

        file_path = f"{dataset_dir}/image_embeddings.pbz2"

        joblib.dump(results, file_path)
        

        
