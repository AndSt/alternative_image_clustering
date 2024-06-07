from typing import List, Dict

from tqdm import tqdm

import os
import random
import multiprocessing

import torch
import numpy as np

from sklearn.model_selection import train_test_split

from clustpy.deep import get_dataloader
from clustpy.deep.autoencoders import FeedforwardAutoencoder
from clustpy.deep import ENRC

from alternative_image_clustering.metrics import get_multiple_labeling_metrics

RANDOM_STATE = 42


def pretrain_autoencoder(
    full_data,
    save_dir: str = None,
    n_epochs=200,
    batch_size=128,
    learning_rate: float = 1e-3,
    device=None,
    random_state=42,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrained_model_path = os.path.join(save_dir, "pretrained_autoencoder.pth")

    model = FeedforwardAutoencoder(layers=[full_data.shape[1], 1024, 512, 256, 20])

    if os.path.exists(pretrained_model_path):
        sd = torch.load(pretrained_model_path)
        model.load_state_dict(sd)
        model.eval()

        model.to(device)
        return model, pretrained_model_path

    model.to(device)
    # The size of the mini-batch that is passed in each trainings iteration
    # Set device on which the model should be trained on (For most of you this will be the CPU)

    data_train, data_eval = train_test_split(
        full_data, test_size=0.2, random_state=random_state
    )

    # create a Dataloader to train the autoencoder in mini-batch fashion
    trainloader = get_dataloader(
        data_train, batch_size=batch_size, shuffle=True, drop_last=True
    )
    evalloader = get_dataloader(
        data_eval, batch_size=batch_size, shuffle=False, drop_last=False
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = {"factor": 0.9, "patience": 5, "verbose": True}

    model.fit(
        n_epochs=n_epochs,
        optimizer_params={"lr": learning_rate},
        dataloader=trainloader,
        evalloader=evalloader,
        device=device,
        model_path=pretrained_model_path,
        scheduler=scheduler,
        scheduler_params=scheduler_params,
        print_step=5,
    )

    return model, pretrained_model_path


def run_single_enrc(
    data,
    labels,
    n_clusters,
    base_dir,
    random_state,
    n_epochs_pretrain=200,
    batch_size_pretrain=128,
    learning_rate=1e-4,
    batch_size: int = 128,
    n_epochs: int = 400,
    step_size: int = 40,
    gamma: float = 0.9
):
    """Function to be executed by each worker process."""

    model, pretrained_model_path = pretrain_autoencoder(
        full_data=data,
        save_dir=base_dir,
        n_epochs=n_epochs_pretrain,
        batch_size=batch_size_pretrain,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    categories = list(n_clusters.keys())

    scheduler = torch.optim.lr_scheduler.StepLR
    scheduler_params = {"step_size": step_size, "gamma": gamma}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enrc = ENRC(
        n_clusters=list(n_clusters.values()),
        batch_size=batch_size,
        # Reduce learning_rate by factor of 10 as usually done in Deep Clustering for stable training
        clustering_optimizer_params={"lr": learning_rate * 0.1},
        clustering_epochs=n_epochs,
        autoencoder=model,
        # Use nrkmeans to initialize ENRC
        init="nrkmeans",
        # Use a random subsample of the data to speed up the initialization procedure
        init_subsample_size=10000,
        device=device,
        scheduler=scheduler,
        scheduler_params=scheduler_params,
        debug=True,
        random_state=random_state,
        recluster=False
    )

    enrc.fit(data)

    pred_clusterings = enrc.labels_

    return {
        "labels": pred_clusterings,
        "cost": 0,  # model.calculate_cost_function(),
        "metrics": get_multiple_labeling_metrics(
            labels_true=labels,
            labels_pred=pred_clusterings,
            categories=categories,
        ),
        "random_state": random_state,
    }


def enrc(
    data: np.ndarray,
    labels: np.ndarray,
    n_clusters: Dict[str, int],
    save_dir: str,
    num_runs=10,
):

    random.seed(RANDOM_STATE)
    random_states = random.sample(range(1000), num_runs)

    os.makedirs(save_dir, exist_ok=True)

    categories = list(n_clusters.keys())
    parameters = [
        (data, labels, n_clusters, save_dir, random_state) for random_state in random_states
    ]

    runs = []
    for parameter in parameters:
        try:
            run = run_single_enrc(*parameter)
            runs.append(run)
        except Exception as e:
            print(f"Error in {parameter}")
            print(e)
            
    if len(runs) < 6:
        raise ValueError("Not enough runs")

    best_run = min(runs, key=lambda x: x["cost"])
    info = {
        "best_cost": best_run["cost"],
        "best_labels": best_run["labels"],
        "best_random_state": best_run["random_state"],
    }
    info["metrics"] = {
        category: {
            metric: np.mean([run["metrics"][category][metric] for run in runs])
            for metric in runs[0]["metrics"][category]
        }
        for category in categories
    }
    info["metrics_stddev"] = {
        category: {
            metric: np.std([run["metrics"][category][metric] for run in runs])
            for metric in runs[0]["metrics"][category]
        }
        for category in categories
    }
    return info
