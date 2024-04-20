from hashlib import sha256
from os.path import exists

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from dataloader import Dataset


def embed(data, embedding_fn="sbert"):
    if embedding_fn == "tfidf":
        num_samples = len(data)
        data_flatten = [sentence for sentences in data for sentence in sentences]
        data_embedded = tfidf(data_flatten).toarray()
        return data_embedded.reshape((num_samples, -1, data_embedded.shape[1]))

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    data_embedded = np.array(list(map(model.encode, tqdm(data, desc="Embedding"))))
    return data_embedded


def embed_dataset(dataset: Dataset, embedding_fn="sbert"):
    cached_results = _load_from_disk(dataset)
    if cached_results is not None:
        # print("Embeddings loaded from disk.")
        return cached_results

    embedding = embed(dataset.answers, embedding_fn)
    _save_to_disk(dataset, embedding)
    return embedding


def tfidf(answers):
    return TfidfVectorizer(stop_words="english").fit_transform(answers)


def sbert(answers):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    embeddings = model.encode(answers)
    return embeddings


def _hash_dataset(dataset):
    return sha256(dataset.answers).hexdigest()


def _save_to_disk(dataset, embedding):
    hash = _hash_dataset(dataset)

    np.save(f"embedding_cache/{hash}.npy", embedding)


def _load_from_disk(dataset):
    hash = _hash_dataset(dataset)
    file_name = f"embedding_cache/{hash}.npy"
    
    if not exists(file_name):
        return None
    
    return np.load(file_name)
