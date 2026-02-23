import os
import sys
import json
import argparse
import time
import torch
import numpy as np

from config import DEFAULT_CONFIG

from src.data_preprocessing  import (
    IoTDataPreprocessor, create_synthetic_iot_data,
    load_bot_iot, load_ton_iot,
)
from src.graph_construction   import IoTGraphConstructor
from src.gnn_models           import get_model, count_parameters
from src.train                import GNNTrainer, compare_models
from src.visualizations       import generate_all_visualizations
from src.explainability       import explain_model
from src.tech_report          import generate_report, save_report


BOLD = "\033[1m"; END = "\033[0m"
CYAN = "\033[96m"; GREEN = "\033[92m"; YELLOW = "\033[93m"


def banner():
    print(f"\n{BOLD}{'═'*65}{END}")
    print(f"{BOLD}{'':^5}IoT Anomaly Detection  |  GNN-based NIDS{END}")
    print(f"{BOLD}{'':^5}E-GraphSAGE + GCN + GAT + GraphSAGE + Hybrid{END}")
    print(f"{BOLD}{'═'*65}{END}\n")


def section(title):
    print(f"\n{CYAN}{BOLD}┌─ {title} {'─'*(55-len(title))}{END}")


def ok(msg):
    print(f"{GREEN}{BOLD}[OK]{END} {msg}")


def info(msg):
    print(f"{CYAN}[INFO]{END} {msg}")


def parse_args():
    p = argparse.ArgumentParser(
        description="IoT Anomaly Detection — GNN Pipeline"
    )
    p.add_argument("--config",      type=str, help="Path to JSON config file")
    p.add_argument("--dataset",     type=str, choices=["synthetic","bot_iot","ton_iot"])
    p.add_argument("--data_path",   type=str)
    p.add_argument("--models",      nargs="+", help="Models to train")
    p.add_argument("--graph",       type=str, choices=["flow","knn","temporal","hybrid"])
    p.add_argument("--epochs",      type=int)
    p.add_argument("--no-explain",  action="store_true", help="Skip XAI step")
    p.add_argument("--multiclass",  action="store_true", help="Multi-class classification")
    return p.parse_args()


def build_config(args) -> dict:
    cfg = DEFAULT_CONFIG.copy()

    if args.config and os.path.isfile(args.config):
        with open(args.config) as f:
            cfg.update(json.load(f))
        info(f"Loaded config from {args.config}")

    overrides = {
        "dataset":      args.dataset,
        "data_path":    args.data_path,
        "graph_method": args.graph,
        "epochs":       args.epochs,
        "multiclass":   args.multiclass or None,
    }
    if args.models:
        cfg["models"] = args.models
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v

    if args.no_explain:
        cfg["explain"] = False

    return cfg


def step_load_data(cfg: dict):
    section("Step 1 — Data Loading & Generation")
    ds = cfg["dataset"]

    if ds == "synthetic":
        df = create_synthetic_iot_data(
            n_samples    = cfg["n_samples"],
            n_devices    = cfg["n_devices"],
            anomaly_ratio= cfg["anomaly_ratio"],
            multiclass   = cfg["multiclass"],
            seed         = cfg["seed"],
        )

    elif ds == "bot_iot":
        path = os.path.join(cfg["data_path"], "bot_iot.csv")
        df = load_bot_iot(path, multiclass=cfg["multiclass"])

    elif ds == "ton_iot":
        path = os.path.join(cfg["data_path"], "ton_iot.csv")
        df = load_ton_iot(path, multiclass=cfg["multiclass"])

    else:
        raise ValueError(f"Unknown dataset: {ds}")

    return df


def step_preprocess(df, cfg: dict):
    section("Step 2 — Preprocessing")
    preprocessor = IoTDataPreprocessor()
    features, labels = preprocessor.fit_transform(df)

    if cfg.get("save_models"):
        os.makedirs(cfg["save_dir"], exist_ok=True)
        preprocessor.save(os.path.join(cfg["save_dir"], "preprocessor.pkl"))

    return features, labels, preprocessor


def step_build_graph(df, features, labels, cfg: dict):
    section("Step 3 — Graph Construction")
    gc    = IoTGraphConstructor(method=cfg["graph_method"],
                                 k_neighbors=cfg["k_neighbors"])
    data  = gc.construct(df, features.values, labels)
    data  = gc.add_masks(data, split=cfg["split_ratio"], seed=cfg["seed"])

    n_classes = int(data.y.max().item()) + 1
    info(f"Graph ready | Classes: {n_classes} | "
         f"Nodes: {data.num_nodes} | Edges: {data.edge_index.shape[1]}")

    return data, gc, n_classes


def step_train(data, cfg: dict, n_classes: int, edge_feat_dim: int):
    section("Step 4 — Model Training")
    node_dim = data.x.shape[1]

    models = {}
    for name in cfg["models"]:
        m = get_model(
            name         = name,
            input_dim    = node_dim,
            hidden_dims  = cfg["hidden_dims"],
            output_dim   = n_classes,
            dropout      = cfg["dropout"],
            heads        = cfg["heads"],
            edge_feat_dim= edge_feat_dim,
        )
        info(f"{name:<16} | parameters: {count_parameters(m):,}")
        models[name] = m

    results, comparison_df = compare_models(
        models,
        data,
        epochs   = cfg["epochs"],
        patience = cfg["patience"],
        save_dir = cfg["save_dir"] if cfg.get("save_models") else None,
    )
    return results, comparison_df, models


def step_visualize(results, comparison_df, data, df_raw, cfg: dict):
    section("Step 5 — Visualizations")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generate_all_visualizations(
        results_dict   = results,
        comparison_df  = comparison_df,
        data           = data,
        df_raw         = df_raw,
        viz_dir        = cfg["viz_dir"],
        device         = device,
    )


def step_explain(results, data, feature_names: list, cfg: dict):
    section("Step 6 — XAI / Explainability")
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    xai_dir = os.path.join(cfg["viz_dir"], "xai")

    for name, r in results.items():
        explain_model(
            name          = name,
            model         = r["trainer"].model,
            data          = data,
            feature_names = feature_names,
            save_dir      = xai_dir,
            device        = device,
        )
    ok(f"XAI plots saved → {xai_dir}")


def step_report(results, comparison_df, cfg: dict):
    section("Step 7 — Technical Report")
    report_txt = generate_report(
        results_dict   = results,
        comparison_df  = comparison_df,
        config         = cfg,
    )
    print(report_txt)

    path = os.path.join(cfg["results_dir"], "technical_report.txt")
    save_report(report_txt, path)

    os.makedirs(cfg["results_dir"], exist_ok=True)
    comparison_df.to_csv(
        os.path.join(cfg["results_dir"], "model_comparison.csv"), index=False
    )
    with open(os.path.join(cfg["results_dir"], "config_used.json"), "w") as f:
        json.dump({k: v for k, v in cfg.items() if not callable(v)}, f, indent=2)

    ok(f"Results saved → {cfg['results_dir']}")


def main():
    banner()
    t0   = time.time()
    args = parse_args()
    cfg  = build_config(args)

    for d in [cfg["save_dir"], cfg["results_dir"], cfg["viz_dir"]]:
        os.makedirs(d, exist_ok=True)

    df                          = step_load_data(cfg)
    features, labels, prep      = step_preprocess(df, cfg)
    data, gc, n_classes         = step_build_graph(df, features, labels, cfg)

    edge_feat_dim = features.shape[1]

    results, comparison_df, models = step_train(data, cfg, n_classes, edge_feat_dim)

    if cfg.get("generate_report"):
        step_visualize(results, comparison_df, data, df, cfg)

    if cfg.get("explain"):
        step_explain(results, data, prep.feature_columns, cfg)

    step_report(results, comparison_df, cfg)

    elapsed = time.time() - t0
    print(f"\n{BOLD}{'═'*65}{END}")
    print(f"{GREEN}{BOLD}  Pipeline complete in {elapsed:.1f}s{END}")
    best = comparison_df.loc[comparison_df["F1-Score"].idxmax()]
    print(f"  Best model  : {BOLD}{best['Model']}{END}  |  "
          f"F1={best['F1-Score']:.4f}  AUC={best['AUC']:.4f}")
    print(f"  Results     : {cfg['results_dir']}")
    print(f"  Visuals     : {cfg['viz_dir']}")
    print(f"{BOLD}{'═'*65}{END}\n")


if __name__ == "__main__":
    main()
