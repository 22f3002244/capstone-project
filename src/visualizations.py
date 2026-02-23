import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc
import networkx as nx
import torch
from typing import Optional, Dict, List
from torch_geometric.data import Data

sns.set_theme(style="whitegrid", palette="deep")
PALETTE = ["#2E86AB", "#A23B72", "#06A77D", "#F18F01", "#C73E1D"]


def plot_training_history(history: dict, model_name: str, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training History — {model_name}", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(history["train_losses"], lw=2, color=PALETTE[0], label="Train Loss", alpha=0.85)
    ax.plot(history["val_losses"],   lw=2, color=PALETTE[1], label="Val Loss",   alpha=0.85)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Loss Curves"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(history["train_accs"], lw=2, color=PALETTE[2], label="Train Acc", alpha=0.85)
    ax.plot(history["val_accs"],   lw=2, color=PALETTE[3], label="Val Acc",   alpha=0.85)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Curves"); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [VIZ] Training history → {save_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    model_name: str,
    save_path: str,
):
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2%", cmap="RdYlGn",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, linecolor="gray", square=True, ax=ax,
        vmin=0, vmax=1,
    )
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j + 0.5, i + 0.72, f"n={cm[i,j]}", ha="center",
                    va="center", fontsize=8, color="#555555", style="italic")

    ax.set_ylabel("True Label",      fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_title(f"Confusion Matrix — {model_name}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [VIZ] Confusion matrix → {save_path}")


def plot_roc_curves(
    roc_data: Dict[str, tuple],
    save_path: str,
):
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, (name, (fpr, tpr, auc_val)) in enumerate(roc_data.items()):
        ax.plot(fpr, tpr, lw=2, color=PALETTE[i % len(PALETTE)],
                label=f"{name}  (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [VIZ] ROC curves → {save_path}")


def compute_roc(model, data: Data, device: str) -> tuple:
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        ea = data.edge_attr if data.edge_attr is not None else None
        out = model(data.x, data.edge_index, ea)
    mask   = data.test_mask
    probs  = torch.softmax(out[mask], dim=1)[:, 1].cpu().numpy()
    true   = data.y[mask].cpu().numpy()
    try:
        fpr, tpr, _ = roc_curve(true, probs)
        auc_val     = auc(fpr, tpr)
    except Exception:
        fpr, tpr, auc_val = np.array([0,1]), np.array([0,1]), 0.5
    return fpr, tpr, auc_val


def plot_metrics_comparison(comparison_df: pd.DataFrame, save_path: str):
    metrics  = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
    n_models = len(comparison_df)
    x        = np.arange(len(metrics))
    width    = 0.8 / n_models

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    ax = axes[0]
    for i, (_, row) in enumerate(comparison_df.iterrows()):
        offset = (i - n_models / 2 + 0.5) * width
        bars   = ax.bar(x + offset, [row[m] for m in metrics],
                        width * 0.9, label=row["Model"],
                        color=PALETTE[i % len(PALETTE)], alpha=0.85)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score"); ax.set_title("Classification Metrics", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9); ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    colors = [PALETTE[i % len(PALETTE)] for i in range(n_models)]
    bars = ax.bar(comparison_df["Model"], comparison_df["Training Time"],
                  color=colors, alpha=0.85)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.3,
                f"{h:.1f}s", ha="center", fontsize=9)
    ax.set_ylabel("Seconds"); ax.set_title("Training Time", fontweight="bold")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Model Comparison Summary", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [VIZ] Metrics comparison → {save_path}")


def plot_radar_chart(comparison_df: pd.DataFrame, save_path: str):
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
    N       = len(metrics)
    angles  = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, (_, row) in enumerate(comparison_df.iterrows()):
        values = [row[m] for m in metrics] + [row[metrics[0]]]
        ax.plot(angles, values, lw=2, color=PALETTE[i % len(PALETTE)],
                label=row["Model"])
        ax.fill(angles, values, color=PALETTE[i % len(PALETTE)], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Model Performance Radar", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [VIZ] Radar chart → {save_path}")


def plot_graph_sample(data: Data, n_nodes: int = 60, save_path: Optional[str] = None):
    G = nx.Graph()
    ei = data.edge_index.cpu().numpy()
    for i in range(ei.shape[1]):
        u, v = int(ei[0, i]), int(ei[1, i])
        if u < n_nodes and v < n_nodes:
            G.add_edge(u, v)

    labels = data.y.cpu().numpy()[:n_nodes]
    pos    = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(10, 7))
    node_colors = ["#ff6b6b" if labels[n] == 1 else "#4ecdc4"
                   for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=80, alpha=0.85, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.7, ax=ax)
    from matplotlib.patches import Patch
    legend = [Patch(color="#4ecdc4", label="Normal"),
              Patch(color="#ff6b6b", label="Anomaly/Attack")]
    ax.legend(handles=legend, loc="upper left", fontsize=10)
    ax.set_title("IoT Communication Graph (sample)", fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  [VIZ] Graph sample → {save_path}")
    plt.close()


def plot_attack_distribution(df: pd.DataFrame, label_col: str, save_path: str):
    counts = df[label_col].value_counts()
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(counts.index, counts.values,
                  color=[PALETTE[i % len(PALETTE)] for i in range(len(counts))],
                  alpha=0.85, edgecolor="white")
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 20,
                f"{b.get_height():,}", ha="center", fontsize=9)
    ax.set_xlabel("Class", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Dataset Label Distribution", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [VIZ] Label distribution → {save_path}")


def generate_all_visualizations(
    results_dict: dict,
    comparison_df: pd.DataFrame,
    data: Data,
    df_raw,
    viz_dir: str,
    device: str = "cpu",
):
    os.makedirs(viz_dir, exist_ok=True)
    print(f"\n[VIZ] Generating visualizations → {viz_dir}")

    label_col = "attack_type" if "attack_type" in df_raw.columns else "label"
    class_names = sorted(df_raw[label_col].unique()) if label_col in df_raw.columns \
                  else ["Normal", "Anomaly"]
    if len(class_names) > 2:
        class_names_cm = class_names
    else:
        class_names_cm = ["Normal", "Anomaly"]

    roc_data = {}
    for name, r in results_dict.items():
        mdir = os.path.join(viz_dir, name)
        os.makedirs(mdir, exist_ok=True)

        plot_training_history(r["history"], name,
                              os.path.join(mdir, "training_history.png"))

        cm = r["test_metrics"].get("confusion_matrix")
        if cm is not None:
            n_c = cm.shape[0]
            cn = ["Normal", "Anomaly"] if n_c == 2 else [str(i) for i in range(n_c)]
            plot_confusion_matrix(cm, cn, name,
                                  os.path.join(mdir, "confusion_matrix.png"))

        try:
            fpr, tpr, auc_val = compute_roc(r["trainer"].model, data, device)
            roc_data[name]    = (fpr, tpr, auc_val)
        except Exception:
            pass

    if roc_data:
        plot_roc_curves(roc_data, os.path.join(viz_dir, "roc_curves.png"))

    plot_metrics_comparison(comparison_df,
                             os.path.join(viz_dir, "model_comparison.png"))
    plot_radar_chart(comparison_df,
                     os.path.join(viz_dir, "radar_chart.png"))

    label_col_dist = "attack_type" if "attack_type" in df_raw.columns else "label"
    if label_col_dist in df_raw.columns:
        plot_attack_distribution(df_raw, label_col_dist,
                                 os.path.join(viz_dir, "label_distribution.png"))

    plot_graph_sample(data, n_nodes=80,
                      save_path=os.path.join(viz_dir, "graph_sample.png"))

    print(f"[VIZ] All visualizations saved.")
