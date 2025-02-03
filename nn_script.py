import numpy as np, os, pandas as pd
import yaml, json, torch.nn as nn, torch
from munch import munchify
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
from sklearn.model_selection import ShuffleSplit, LeaveOneOut

from nn_model import MultimodalNN
from nn_dataset import TRINA33
from nn_trainer import Trainer


def find_class_weights(labels: list[int]) -> torch.Tensor:
    class_weights = np.unique(labels, return_counts=True)[1]
    class_weights = class_weights.sum() / class_weights
    return torch.tensor(class_weights / class_weights.sum()).float()


def task_exists(sub, tasks):
    for task in tasks:
        task_file = f"{sub}_{tasks[0]}.csv"
        if not os.path.exists(cfg.root + task_file):
            print(f"Task {task} does not exist for subject {sub}.")
            return False
    return True


# Load config file
os.makedirs("ckpt", exist_ok=True)
with open("config.yaml", "r") as f:
    cfg = munchify(yaml.safe_load(f))
print("Using config:", cfg)

if "cuda" in cfg.device:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

subs = [f"P{i}" for i in range(1, 34)]
bad_subs = ["P3", "P9", "P25"]
subs = [sub for sub in subs if sub not in bad_subs]

loso = LeaveOneOut()
all_scores = {s: {} for s in subs}
all_cms = {s: {} for s in subs}
for i, (train_index, test_index) in enumerate(loso.split(subs)):
    print(f"\nFold {i+1}/{len(subs)}")
    train_subs, test_sub = np.array(subs)[train_index], np.array(subs)[test_index]
    print(f"Test subject: {test_sub}")

    if not task_exists(test_sub[0], cfg.tasks):
        continue
    if int(test_sub[0][1:]) < 17:
        continue

    ssp = ShuffleSplit(
        n_splits=cfg.n_splits,
        test_size=cfg.test_size,
    )

    these_scores, these_cm = [], []
    for j, (tr_index, vl_index) in enumerate(ssp.split(train_subs)):
        print(f"Seed {j+1}/{cfg.n_splits}")
        tr_sub, val_sub = train_subs[tr_index], train_subs[vl_index]

        print("Loading training dataset...")
        train_dataset = TRINA33(
            cfg.proc_root, task=None, subjects=tr_sub, modalities=cfg.modalities
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=cfg.num_workers,
        )
        tr_mean = train_dataset.mean
        tr_std = train_dataset.std

        print("Loading validation dataset...")
        val_dataset = TRINA33(
            cfg.proc_root,
            task=None,
            subjects=val_sub,
            modalities=cfg.modalities,
            mean=tr_mean,
            std=tr_std,
        )
        valid_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=False,
        )

        # Load the model
        model = MultimodalNN(
            n_modalities=len(cfg.modalities),
            in_channels=cfg.model_config.in_channels,
            base_filters=cfg.model_config.base_filters,
            n_block=cfg.model_config.n_block,
            n_classes=cfg.model_config.n_classes,
        )
        model.to(cfg.device)
        model = nn.DataParallel(model)

        # Define optimizer, scheduler, and loss function
        optimizer = AdamW(model.parameters(), lr=cfg.lr, eps=1e-8, weight_decay=1e-5)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.step_size)

        # Define the trainer
        trainer = Trainer(
            device=cfg.device,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopping=cfg.early_stopping,
            checkpoint=f"ckpt/{test_sub[0]}_seed{j}_task{"".join(cfg.tasks)}.pt",
            patience=cfg.patience,
            epochs=cfg.max_epochs,
            class_weights=find_class_weights(train_dataset.labels),
            verbose=False,
        )
        print("Now training...\n")
        trainer.perform_training(train_loader, valid_loader)

        print("\nNow testing...")
        test_dataset = TRINA33(
            cfg.proc_root,
            task=None,
            subjects=test_sub,
            modalities=cfg.modalities,
            mean=tr_mean,
            std=tr_std,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
        )
        scores_list, cm = trainer.perform_inference(test_loader)
        these_scores.append(scores_list)
        these_cm.append(cm)

    # Average the scores
    all_scores[test_sub[0]] = (
        {
            k: np.array([x[k] for x in these_scores]).mean().tolist()
            for k in these_scores[0]
        }
        if isinstance(these_scores[0], dict)
        else np.array(these_scores).mean(axis=0).tolist()
    )
    print(f"Sum confusion matrix:\n{np.array(these_cm).sum(axis=0)}")
    all_cms[test_sub[0]] = np.array(these_cm).sum(axis=0).tolist()

    # Save the scores as we go
    os.makedirs("results", exist_ok=True)
    with open("results/nn_scores_I2.json", "w") as f:
        json.dump(all_scores, f, indent=4)
    with open("results/nn_scores_I2_cm.json", "w") as f:
        yaml.dump(all_cms, f, indent=4)
