import os
from datetime import datetime
from typing import Dict, Any
import numpy as np
import pandas as pd
import torch


W    = 100
SEP  = "═" * W
THIN = "─" * W


def _bar(v: float, w: int = 28) -> str:
    filled = int(round(v * w))
    return "█" * filled + "░" * (w - filled)


def _badge(v: float) -> str:
    if v >= 0.97: return "◆ EXCELLENT"
    if v >= 0.90: return "● GOOD"
    if v >= 0.80: return "○ ACCEPTABLE"
    return "△ NEEDS WORK"


def generate_report(
    results_dict: Dict[str, Any],
    comparison_df: pd.DataFrame,
    config: Dict[str, Any] = None,
    class_names: list = None,
) -> str:
    cfg = config or {}
    now = datetime.now()
    r   = []

    def line(s=""):
        r.append(s)

    line(SEP)
    title = "IOT ANOMALY DETECTION — TECHNICAL EVALUATION REPORT"
    pad   = (W - len(title)) // 2
    line(" " * pad + title)
    line(SEP)
    line(f"  Generated  : {now.strftime('%Y-%m-%d  %H:%M:%S')}")
    line(f"  Report ID  : IADP-{now.strftime('%Y%m%d-%H%M%S')}")
    line(f"  Framework  : E-GraphSAGE + GCN + GAT + GraphSAGE + Hybrid  |  PyTorch Geometric")
    line(f"  Reference  : Lo et al. (IEEE/IFIP NOMS 2022) — E-GraphSAGE NIDS")
    line(SEP)
    line()

    line("  SECTION 1 — EXPERIMENT CONFIGURATION")
    line(THIN)
    for k, v in [
        ("Dataset",                 cfg.get("dataset", "synthetic")),
        ("Samples",                 f"{cfg.get('n_samples', 12000):,}"),
        ("Devices",                 cfg.get("n_devices", 60)),
        ("Anomaly Ratio",           f"{cfg.get('anomaly_ratio', 0.15):.1%}"),
        ("Classification",          "Multi-class" if cfg.get("multiclass") else "Binary"),
        ("Graph Method",            cfg.get("graph_method", "flow")),
        ("k-Neighbors",             cfg.get("k_neighbors", 7)),
        ("Train / Val / Test",      " / ".join(f"{v:.0%}" for v in cfg.get("split_ratio", [0.7,0.15,0.15]))),
        ("Hidden Dimensions",       str(cfg.get("hidden_dims", [128, 64]))),
        ("Dropout",                 cfg.get("dropout", 0.3)),
        ("Max Epochs",              cfg.get("epochs", 300)),
        ("Early-Stop Patience",     cfg.get("patience", 25)),
        ("Learning Rate",           cfg.get("learning_rate", 0.001)),
        ("Weight Decay",            cfg.get("weight_decay", 5e-4)),
    ]:
        line(f"  {k:<32} {str(v)}")
    line(f"  {'Compute Device':<32} {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
    line()

    line("  SECTION 2 — PER-MODEL RESULTS")
    line(THIN)

    for idx, (model_name, res) in enumerate(results_dict.items(), 1):
        m       = res.get("test_metrics", res)
        history = res.get("history", {})
        tt      = history.get("training_time", 0)
        epochs  = len(history.get("train_losses", []))
        bvl     = history.get("best_val_loss", float("nan"))

        acc   = m.get("accuracy",  0)
        prec  = m.get("precision", 0)
        rec   = m.get("recall",    0)
        f1    = m.get("f1",        0)
        auc_v = m.get("auc",       0)
        loss  = m.get("loss",      0)
        cm    = m.get("confusion_matrix", None)

        is_egraphsage = "egraphsage" in model_name.lower().replace("-","").replace("_","")

        line(f"  ┌─ [{idx}] {model_name}{' ← E-GraphSAGE (IEEE paper)' if is_egraphsage else ''}")
        line(f"  │")
        line(f"  │  Training")
        line(f"  │    Epochs run         : {epochs}")
        line(f"  │    Training time      : {tt:.2f}s")
        line(f"  │    Best val loss      : {bvl:.6f}")
        line(f"  │")
        line(f"  │  Test Metrics")
        line(f"  │    {'Metric':<18} {'Value':>7}   Visual                         Status")
        line(f"  │    {THIN[4:]}")
        for label, val in [("Accuracy",acc),("Precision",prec),("Recall",rec),
                            ("F1-Score",f1),("AUC-ROC",auc_v)]:
            line(f"  │    {label:<18} {val:>7.4f}   [{_bar(val)}]  {_badge(val)}")
        line(f"  │    {'Test Loss':<18} {loss:>7.6f}")
        line(f"  │")

        if cm is not None and cm.shape == (2, 2):
            tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
            total = tn + fp + fn + tp
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            line(f"  │  Confusion Matrix")
            line(f"  │               {'Pred-Normal':>14}   {'Pred-Anomaly':>14}")
            line(f"  │  True-Normal   {tn:>14,}   {fp:>14,}    FPR: {fpr:.4f}")
            line(f"  │  True-Anomaly  {fn:>14,}   {tp:>14,}    FNR: {fnr:.4f}")
            line(f"  │              (n={total:,})")

        line(f"  └{'─'*98}")
        line()

    line("  SECTION 3 — MODEL COMPARISON TABLE")
    line(THIN)

    cols = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC", "Training Time"]
    hdr  = f"  {'Model':<20}" + "".join(f"{c:>12}" for c in cols)
    line(hdr)
    line(f"  {THIN[2:]}")

    best_f1_idx  = comparison_df["F1-Score"].idxmax()
    best_auc_idx = comparison_df["AUC"].idxmax()

    for i, row in comparison_df.iterrows():
        marker = ""
        if i == best_f1_idx and i == best_auc_idx:
            marker = "  ◀ BEST F1 & AUC"
        elif i == best_f1_idx:
            marker = "  ◀ BEST F1"
        elif i == best_auc_idx:
            marker = "  ◀ BEST AUC"
        vals = (f"  {row['Model']:<20}"
                f"{row['Accuracy']:>12.4f}"
                f"{row['Precision']:>12.4f}"
                f"{row['Recall']:>12.4f}"
                f"{row['F1-Score']:>12.4f}"
                f"{row['AUC']:>12.4f}"
                f"{row['Training Time']:>12.2f}"
                f"{marker}")
        line(vals)
    line()

    line("  SECTION 4 — STATISTICAL SUMMARY")
    line(THIN)
    stat_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
    line(f"  {'Metric':<18} {'Mean':>8}   {'Std':>8}   {'Min':>8}   {'Max':>8}")
    line(f"  {THIN[2:]}")
    for col in stat_cols:
        d = comparison_df[col]
        line(f"  {col:<18} {d.mean():>8.4f}   {d.std():>8.4f}   {d.min():>8.4f}   {d.max():>8.4f}")
    line()

    line("  SECTION 5 — E-GRAPHSAGE ANALYSIS (vs. Baselines)")
    line(THIN)

    eg_rows = comparison_df[comparison_df["Model"].str.lower().str.replace("-","").str.replace("_","")
                            .str.contains("egraphsage")]
    baseline_rows = comparison_df[~comparison_df.index.isin(eg_rows.index)]

    if not eg_rows.empty and not baseline_rows.empty:
        eg = eg_rows.iloc[0]
        for col in ["F1-Score", "AUC", "Accuracy"]:
            best_base_val  = baseline_rows[col].max()
            best_base_name = baseline_rows.loc[baseline_rows[col].idxmax(), "Model"]
            delta = eg[col] - best_base_val
            sign  = "+" if delta >= 0 else ""
            line(f"  {col:<18}: E-GraphSAGE={eg[col]:.4f}  |  "
                 f"Best Baseline={best_base_val:.4f} ({best_base_name})  |  "
                 f"Δ={sign}{delta:.4f}")
        line()
        line(f"  E-GraphSAGE key advantage: edge-level flow classification allows")
        line(f"  detection of individual malicious flows, not just device-level anomalies.")
        line(f"  This aligns with the IoT NIDS requirement to identify specific attack flows.")
    else:
        line("  [Note: Run with EGraphSAGE in the models list to see this comparison.]")
    line()

    line("  SECTION 6 — RECOMMENDATIONS")
    line(THIN)

    best  = comparison_df.loc[comparison_df["F1-Score"].idxmax()]
    worst = comparison_df.loc[comparison_df["F1-Score"].idxmin()]
    fast  = comparison_df.loc[comparison_df["Training Time"].idxmin()]
    sec_best_idx = comparison_df["F1-Score"].nlargest(2).index
    sec_best = comparison_df.loc[[i for i in sec_best_idx if i != comparison_df["F1-Score"].idxmax()][0]]

    line(f"  [REC-1]  PRIMARY DEPLOYMENT → {best['Model']}")
    line(f"           F1={best['F1-Score']:.4f}  AUC={best['AUC']:.4f}  Acc={best['Accuracy']:.4f}")
    line()
    line(f"  [REC-2]  BACKUP / ENSEMBLE  → {sec_best['Model']}  (F1={sec_best['F1-Score']:.4f})")
    line(f"           Consider ensemble voting for higher confidence decisions.")
    line()
    line(f"  [REC-3]  LIGHTWEIGHT OPTION → {fast['Model']}  ({fast['Training Time']:.1f}s)")
    line(f"           Use on resource-constrained IoT gateways.")
    line()
    line(f"  [REC-4]  THRESHOLD TUNING")
    line(f"           Default decision threshold = 0.5.")
    line(f"           → Lower threshold: higher recall, more false positives (critical environments).")
    line(f"           → Higher threshold: fewer false alarms (operational environments).")
    line()
    line(f"  [REC-5]  PERIODIC RETRAINING")
    line(f"           IoT traffic distributions drift over time.  Retrain every 30–90 days")
    line(f"           or when F1 on held-out data drops below 0.90.")
    line()
    line(f"  [REC-6]  REAL DATASET EVALUATION")
    line(f"           Validate on BoT-IoT / ToN-IoT / NF-BoT-IoT benchmark datasets")
    line(f"           (set dataset=bot_iot or ton_iot in config).")
    line()
    line(f"  [MONITOR] {worst['Model']} has lowest F1={worst['F1-Score']:.4f}.")
    line(f"            Consider removing if performance does not improve after retraining.")
    line()

    line(SEP)
    line(f"  END OF REPORT  |  {now.strftime('%Y-%m-%d  %H:%M:%S')}")
    line(f"  Reference: Lo, W.W. et al. E-GraphSAGE (IEEE/IFIP NOMS 2022)")
    line(SEP)

    return "\n".join(r)


def save_report(content: str, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[OK] Technical report saved → {path}")
