import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from typing import Optional, List, Dict
from torch_geometric.data import Data

from src.gnn_models import EGraphSAGE, GAT


class GradientExplainer:

    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model  = model.to(device)
        self.device = device

    def explain(
        self,
        data: Data,
        target_class: int = 1,
        n_samples: int = 50,
    ) -> np.ndarray:
        self.model.eval()
        data = data.to(self.device)

        x        = data.x.clone().requires_grad_(True)
        edge_idx = data.edge_index
        edge_a   = data.edge_attr if data.edge_attr is not None else None

        out = self.model(x, edge_idx, edge_a)

        mask  = data.test_mask
        idx   = (data.y[mask] == target_class).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            return np.zeros(x.shape[1])

        sample_idx = idx[:n_samples]
        loss = -F.log_softmax(out[mask][sample_idx], dim=1)[:, target_class].mean()
        loss.backward()

        with torch.no_grad():
            saliency = (x.grad.abs() * x.abs()).mean(dim=0)
            saliency = (saliency / (saliency.sum() + 1e-8)).cpu().numpy()

        return saliency

    def plot_feature_importance(
        self,
        saliency: np.ndarray,
        feature_names: List[str],
        top_k: int = 15,
        save_path: Optional[str] = None,
        title: str = "Feature Importance (Gradient × Input)",
    ):
        k   = min(top_k, len(saliency))
        idx = np.argsort(saliency)[-k:][::-1]
        vals = saliency[idx]
        names = [feature_names[i] if i < len(feature_names) else f"f{i}" for i in idx]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(range(k), vals[::-1],
                       color=plt.cm.RdYlGn(vals[::-1] / (vals.max() + 1e-8)))
        ax.set_yticks(range(k))
        ax.set_yticklabels(names[::-1], fontsize=10)
        ax.set_xlabel("Normalised Importance", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"  [XAI] Feature importance plot → {save_path}")
        plt.close()


class AttentionExplainer:

    def __init__(self, model, device: str = "cpu"):
        if not isinstance(model, GAT):
            raise TypeError("AttentionExplainer requires a GAT model.")
        self.model  = model.to(device)
        self.device = device
        self._attn_weights = []

    def _hook(self, module, inp, out):
        if isinstance(out, tuple) and len(out) == 2:
            self._attn_weights.append(out[1].detach())

    def explain(self, data: Data) -> Dict[str, np.ndarray]:
        data = data.to(self.device)
        self._attn_weights.clear()

        handles = []
        for conv in self.model.convs:
            handles.append(conv.register_forward_hook(self._hook))

        self.model.eval()
        with torch.no_grad():
            self.model(data.x, data.edge_index)

        for h in handles:
            h.remove()

        results = {}
        for i, aw in enumerate(self._attn_weights):
            results[f"layer_{i+1}"] = aw.mean(-1).cpu().numpy()
        return results

    def plot_attention_distribution(
        self,
        attn_dict: Dict[str, np.ndarray],
        save_path: Optional[str] = None,
    ):
        n = len(attn_dict)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
        if n == 1:
            axes = [axes]

        for ax, (layer_name, weights) in zip(axes, attn_dict.items()):
            ax.hist(weights, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
            ax.set_title(f"Attention — {layer_name}", fontweight="bold")
            ax.set_xlabel("Attention weight")
            ax.set_ylabel("Count")
            ax.grid(alpha=0.3)

        plt.suptitle("GAT Attention Weight Distribution", fontsize=13, fontweight="bold")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"  [XAI] Attention distribution → {save_path}")
        plt.close()


class EdgeImportanceExplainer:

    def __init__(self, model: EGraphSAGE, device: str = "cpu"):
        if not isinstance(model, EGraphSAGE):
            raise TypeError("EdgeImportanceExplainer requires an EGraphSAGE model.")
        self.model  = model.to(device)
        self.device = device

    @torch.no_grad()
    def score_edges(self, data: Data) -> np.ndarray:
        self.model.eval()
        data = data.to(self.device)
        logits = self.model(data.x, data.edge_index, data.edge_attr)
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        return probs

    @torch.no_grad()
    def feature_perturbation(
        self,
        data: Data,
        n_edges: int = 200,
    ) -> np.ndarray:
        self.model.eval()
        data = data.to(self.device)

        base_probs  = self.score_edges(data)
        attack_mask = base_probs > 0.5
        n_feat      = data.edge_attr.shape[1]
        importance  = np.zeros(n_feat)

        for f in range(n_feat):
            ea_perturbed = data.edge_attr.clone()
            ea_perturbed[:, f] = 0.0
            logits = self.model(data.x, data.edge_index, ea_perturbed)
            perturbed_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            if attack_mask.sum() > 0:
                importance[f] = (base_probs[attack_mask] - perturbed_probs[attack_mask]).mean()

        importance = np.clip(importance, 0, None)
        importance /= (importance.sum() + 1e-8)
        return importance

    def plot_edge_risk(
        self,
        probs: np.ndarray,
        true_labels: np.ndarray,
        save_path: Optional[str] = None,
    ):
        fig, ax = plt.subplots(figsize=(9, 5))
        normal_p = probs[true_labels == 0]
        attack_p = probs[true_labels == 1]
        ax.hist(normal_p, bins=60, alpha=0.7, color="#4ecdc4", label="Normal", density=True)
        ax.hist(attack_p, bins=60, alpha=0.7, color="#ff6b6b", label="Attack", density=True)
        ax.axvline(0.5, color="k", linestyle="--", lw=1.5, label="Threshold 0.5")
        ax.set_xlabel("Attack Probability", fontsize=11)
        ax.set_ylabel("Density",            fontsize=11)
        ax.set_title("E-GraphSAGE: Edge Risk Score Distribution",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"  [XAI] Edge risk plot → {save_path}")
        plt.close()

    def plot_feature_importance(
        self,
        importance: np.ndarray,
        feature_names: List[str],
        save_path: Optional[str] = None,
    ):
        k    = len(importance)
        idx  = np.argsort(importance)[::-1]
        vals = importance[idx]
        names = [feature_names[i] if i < len(feature_names) else f"f{i}" for i in idx]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.Reds(vals / (vals.max() + 1e-8))
        ax.barh(range(k), vals[::-1], color=colors[::-1])
        ax.set_yticks(range(k))
        ax.set_yticklabels(names[::-1], fontsize=9)
        ax.set_xlabel("Perturbation Importance", fontsize=11)
        ax.set_title("E-GraphSAGE: Edge Feature Importance",
                     fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"  [XAI] E-GraphSAGE feature importance → {save_path}")
        plt.close()


def explain_model(
    name: str,
    model,
    data: Data,
    feature_names: List[str],
    save_dir: str,
    device: str = "cpu",
):
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n  [XAI] Explaining {name}...")

    if isinstance(model, EGraphSAGE):
        exp    = EdgeImportanceExplainer(model, device)
        probs  = exp.score_edges(data)
        true_l = data.y.cpu().numpy()
        exp.plot_edge_risk(
            probs, true_l,
            os.path.join(save_dir, f"{name}_edge_risk.png")
        )
        importance = exp.feature_perturbation(data)
        exp.plot_feature_importance(
            importance, feature_names,
            os.path.join(save_dir, f"{name}_feature_importance.png")
        )

    elif isinstance(model, GAT):
        try:
            exp   = AttentionExplainer(model, device)
            attn  = exp.explain(data)
            exp.plot_attention_distribution(
                attn,
                os.path.join(save_dir, f"{name}_attention.png")
            )
        except Exception as e:
            print(f"    [WARN] GAT attention extraction failed: {e}")

    try:
        gexp     = GradientExplainer(model, device)
        saliency = gexp.explain(data, target_class=1)
        gexp.plot_feature_importance(
            saliency, feature_names,
            save_path=os.path.join(save_dir, f"{name}_gradient_saliency.png"),
            title=f"{name}: Gradient Saliency (Attack Class)",
        )
    except Exception as e:
        print(f"    [WARN] Gradient explainer failed for {name}: {e}")
