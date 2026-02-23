DEFAULT_CONFIG = {
    "dataset": "synthetic",
    "data_path": "data/",
    "n_samples": 12000,
    "n_devices": 60,
    "anomaly_ratio": 0.15,
    "multiclass": False,
    "seed": 42,

    "graph_method": "flow",
    "k_neighbors": 7,
    "split_ratio": [0.70, 0.15, 0.15],

    "models": ["EGraphSAGE", "GCN", "GAT", "GraphSAGE", "Hybrid"],

    "hidden_dims": [128, 64],
    "dropout": 0.3,
    "heads": 4,

    "epochs": 300,
    "patience": 25,
    "learning_rate": 0.001,
    "weight_decay": 5e-4,
    "batch_size": 512,

    "save_dir": "models/",
    "results_dir": "results/",
    "viz_dir": "visualizations/",
    "save_models": True,
    "generate_report": True,

    "explain": True,
    "n_explain_samples": 5,
}
