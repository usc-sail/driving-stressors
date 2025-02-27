import numpy as np, torch
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
)
from scipy.special import softmax


class Trainer:
    def __init__(
        self,
        device: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        early_stopping: bool,
        checkpoint: str,
        patience: int,
        epochs: int,
        class_weights: torch.Tensor,
        verbose: bool = True,
    ) -> None:
        self.device = device
        self.model = model
        self.model.to(self.device)
        self.checkpoint = checkpoint
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.class_weights = class_weights.to(self.device)
        self.criterion = CrossEntropyLoss(weight=self.class_weights)
        self.early_stopping = early_stopping
        self.patience = patience
        self.epochs = epochs
        self.verbose = verbose

    def perform_training(
        self, train_loader: DataLoader, valid_loader: DataLoader
    ) -> None:
        best_metric = 10
        patience_counter = 0
        for ep in range(self.epochs):
            self.train(train_loader, ep)

            metric_dict, _, _ = self.validate(valid_loader, ep)
            if self.verbose:
                for metric in metric_dict:
                    print(f"\tvalid {metric}: {metric_dict[metric]:.7f}")
                print()

            if metric_dict["CE Loss"] < best_metric:
                best_metric = metric_dict["CE Loss"]
                patience_counter = 0
                self.save(self.checkpoint)
            else:
                patience_counter += 1
                if self.early_stopping and patience_counter == self.patience:
                    print("Early stopping now.")
                    break
            self.scheduler.step(metric_dict["CE Loss"])

    def perform_inference(self, data_loader: DataLoader) -> dict[str, float]:
        self.load(self.checkpoint)
        self.metric_dict, all_proba, cm = self.validate(data_loader)
        for metric in self.metric_dict:
            print(f"\ttest {metric}: {self.metric_dict[metric]:.7f}")
        return all_proba, cm

    def train(self, data_loader: DataLoader, epoch) -> float:
        self.model.train()
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}", leave=False)

        total_loss = 0
        for samples, labels, _ in progress_bar:
            self.optimizer.zero_grad()

            samples = samples.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(samples)

            loss = self.criterion(outputs.softmax(1), labels.long())
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            progress_bar.set_postfix({"loss": f"{loss.item():.3f}"})

    def validate(
        self, data_loader: DataLoader, epoch: int | None = None
    ) -> tuple[dict[str, float], dict[str, float]]:
        self.model.eval()
        disp = epoch + 1 if epoch is not None else "TEST"
        progress_bar = tqdm(data_loader, desc=f"Epoch {disp}", leave=False)

        all_pred, all_true, all_name, all_loss = [], [], [], []
        with torch.no_grad():
            for samples, labels, subjects in progress_bar:

                samples = samples.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(samples)
                loss = self.criterion(outputs.softmax(1), labels.long())

                all_pred.extend(outputs.cpu().numpy())
                all_true.extend(labels.cpu().numpy())
                all_name.extend(subjects)
                all_loss.append(loss.item())

        all_pred = np.array(all_pred)
        all_true = np.array(all_true).squeeze()
        all_proba = softmax(all_pred, axis=1)[:, 1]
        all_pred_m = np.argmax(all_pred, axis=1)

        # confusion matrix
        n_labels = self.class_weights.shape[0]
        cm = confusion_matrix(all_true, all_pred_m, labels=np.arange(n_labels))

        score_dict = {
            "CE Loss": np.mean(all_loss),
            "ROC-AUC": roc_auc_score(all_true, all_proba),
            "F1 Macro": f1_score(all_true, all_pred_m, average="macro"),
            "B. Accuracy": balanced_accuracy_score(all_true, all_pred_m),
        }
        return score_dict, all_proba, cm

    def save(self, file_path: str = "./") -> None:
        torch.save(self.model.state_dict(), file_path)

    def load(self, file_path: str = "./") -> None:
        self.model.load_state_dict(torch.load(file_path))
        self.model.to(self.device)
