import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report,
)
from tqdm import tqdm
from typing import Dict, Optional
import pandas as pd

from src.gnn_models import EGraphSAGE, count_parameters


def _is_edge_model(model: nn.Module) -> bool:
    return isinstance(model, EGraphSAGE)


def _compute_class_weights(labels: torch.Tensor, n_classes: int,
                            device) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=n_classes).float()
    counts = torch.where(counts == 0, torch.ones_like(counts), counts)
    w = 1.0 / counts
    return (w / w.sum() * n_classes).to(device)


class GNNTrainer:

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        weight_decay: float  = 5e-4,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = model.to(self.device)
        self.opt    = optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode="min", factor=0.5, patience=10, verbose=False
        )

        self.train_losses: list = []
        self.val_losses:   list = []
        self.train_accs:   list = []
        self.val_accs:     list = []
        self.best_val_loss: float = float("inf")
        self._best_state   = None
        self._edge_model   = _is_edge_model(model)

    def _forward(self, data: Data):
        x   = data.x.to(self.device)
        ei  = data.edge_index.to(self.device)
        ea  = data.edge_attr.to(self.device) if data.edge_attr is not None else None
        out = self.model(x, ei, ea)
        if self._edge_model:
            y = data.y.to(self.device)
        else:
            y = data.node_y.to(self.device)
        return out, y

    def _masked(self, data: Data, mask_name: str):
        out, y = self._forward(data)
        if self._edge_model:
            mask = getattr(data, mask_name).to(self.device)
        else:
            mask = getattr(data, "node_" + mask_name).to(self.device)
        return out[mask], y[mask]

    def train_epoch(self, data: Data, criterion: nn.Module) -> tuple:
        self.model.train()
        self.opt.zero_grad()
        logits, targets = self._masked(data, "train_mask")
        loss = criterion(logits, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        acc = (logits.argmax(1) == targets).float().mean().item()
        return loss.item(), acc

    @torch.no_grad()
    def evaluate(self, data: Data, mask_name: str, criterion: nn.Module) -> dict:
        self.model.eval()
        logits, targets = self._masked(data, mask_name)
        loss = criterion(logits, targets).item()

        pred_np  = logits.argmax(1).cpu().numpy()
        true_np  = targets.cpu().numpy()
        probs_np = torch.softmax(logits, dim=1).cpu().numpy()

        n_cls = logits.shape[1]
        avg   = "binary" if n_cls == 2 else "weighted"

        metrics = {
            "loss":      loss,
            "accuracy":  accuracy_score(true_np, pred_np),
            "precision": precision_score(true_np, pred_np, average=avg, zero_division=0),
            "recall":    recall_score(   true_np, pred_np, average=avg, zero_division=0),
            "f1":        f1_score(       true_np, pred_np, average=avg, zero_division=0),
        }

        if n_cls == 2:
            uniq = np.unique(true_np)
            if len(uniq) == 2:
                metrics["auc"] = roc_auc_score(true_np, probs_np[:, 1])
            else:
                metrics["auc"] = 0.5
        else:
            try:
                metrics["auc"] = roc_auc_score(
                    true_np, probs_np, multi_class="ovr", average="weighted"
                )
            except Exception:
                metrics["auc"] = 0.0

        return metrics

    def train(
        self,
        data: Data,
        epochs: int  = 300,
        patience: int = 25,
        verbose: bool = True,
        save_dir: Optional[str] = None,
    ) -> dict:
        data = data.to(self.device)

        if self._edge_model:
            train_labels = data.y[data.train_mask]
        else:
            train_labels = data.node_y[data.node_train_mask]
        n_cls    = int(train_labels.max().item()) + 1
        weights  = _compute_class_weights(train_labels, n_cls, self.device)
        criterion = nn.CrossEntropyLoss(weight=weights)

        if verbose:
            print(f"\n{'─'*65}")
            print(f"  Model      : {self.model.__class__.__name__}")
            print(f"  Parameters : {count_parameters(self.model):,}")
            print(f"  Device     : {self.device}")
            print(f"  Classes    : {n_cls}")
            print(f"  Max epochs : {epochs}  |  Patience: {patience}")
            print(f"{'─'*65}")

        patience_ctr = 0
        start        = time.time()
        itr          = tqdm(range(epochs), desc=f"  {self.model.__class__.__name__:<14}",
                            unit="ep", leave=True) if verbose else range(epochs)

        for epoch in itr:
            t_loss, t_acc = self.train_epoch(data, criterion)
            v_metrics     = self.evaluate(data, "val_mask", criterion)
            v_loss, v_acc = v_metrics["loss"], v_metrics["accuracy"]

            self.train_losses.append(t_loss)
            self.val_losses.append(v_loss)
            self.train_accs.append(t_acc)
            self.val_accs.append(v_acc)

            self.scheduler.step(v_loss)

            if verbose and isinstance(itr, tqdm):
                itr.set_postfix({
                    "tL": f"{t_loss:.3f}",
                    "vL": f"{v_loss:.3f}",
                    "vF1": f"{v_metrics['f1']:.3f}",
                    "vAUC": f"{v_metrics['auc']:.3f}",
                })

            if v_loss < self.best_val_loss:
                self.best_val_loss = v_loss
                self._best_state   = copy.deepcopy(self.model.state_dict())
                patience_ctr       = 0
                if save_dir:
                    self._save_ckpt(save_dir, epoch, v_metrics)
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    if verbose:
                        print(f"\n  Early stop at epoch {epoch + 1}")
                    break

        if self._best_state:
            self.model.load_state_dict(self._best_state)

        elapsed = time.time() - start
        if verbose:
            print(f"\n  Done in {elapsed:.1f}s  |  best val loss: {self.best_val_loss:.4f}")

        return {
            "train_losses": self.train_losses,
            "val_losses":   self.val_losses,
            "train_accs":   self.train_accs,
            "val_accs":     self.val_accs,
            "training_time": elapsed,
            "best_val_loss": self.best_val_loss,
        }

    def test(self, data: Data) -> dict:
        data = data.to(self.device)
        if self._edge_model:
            test_labels = data.y[data.test_mask]
        else:
            test_labels = data.node_y[data.node_test_mask]
        n_cls     = int(test_labels.max().item()) + 1
        weights   = _compute_class_weights(test_labels, n_cls, self.device)
        criterion = nn.CrossEntropyLoss(weight=weights)

        metrics = self.evaluate(data, "test_mask", criterion)

        self.model.eval()
        with torch.no_grad():
            logits, targets = self._masked(data, "test_mask")
        pred = logits.argmax(1).cpu().numpy()
        true = targets.cpu().numpy()

        metrics["confusion_matrix"]      = confusion_matrix(true, pred)
        metrics["classification_report"] = classification_report(
            true, pred, output_dict=True, zero_division=0
        )
        return metrics

    def _save_ckpt(self, save_dir: str, epoch: int, metrics: dict):
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            "epoch":     epoch,
            "state":     self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "metrics":   metrics,
        }, os.path.join(save_dir, "best_model.pth"))

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["state"])
        return ckpt["metrics"]


def compare_models(
    models_dict: Dict[str, nn.Module],
    data: Data,
    epochs: int   = 300,
    patience: int = 25,
    save_dir: Optional[str] = None,
    verbose: bool = True,
) -> tuple:
    results = {}
    for name, model in models_dict.items():
        print(f"\n{'═'*65}")
        print(f"  Training: {name}")
        print(f"{'═'*65}")

        trainer  = GNNTrainer(model)
        mdir     = os.path.join(save_dir, name) if save_dir else None
        history  = trainer.train(data, epochs=epochs, patience=patience,
                                 verbose=verbose, save_dir=mdir)
        test_m   = trainer.test(data)

        results[name] = {
            "history":      history,
            "test_metrics": test_m,
            "trainer":      trainer,
        }

    rows = []
    for name, r in results.items():
        m = r["test_metrics"]
        rows.append({
            "Model":         name,
            "Accuracy":      m["accuracy"],
            "Precision":     m["precision"],
            "Recall":        m["recall"],
            "F1-Score":      m["f1"],
            "AUC":           m["auc"],
            "Training Time": r["history"]["training_time"],
        })

    df = pd.DataFrame(rows)
    print(f"\n{'═'*65}\n  Model Comparison\n{'─'*65}")
    print(df.to_string(index=False, float_format="%.4f"))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(os.path.join(save_dir, "model_comparison.csv"), index=False)

    return results, df
