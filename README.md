# IoT Anomaly Detection using Graph Neural Networks

A production-quality pipeline for detecting network intrusions in IoT environments using Graph Neural Networks. Implements **E-GraphSAGE** (Lo et al., IEEE/IFIP NOMS 2022) as the primary model, alongside GCN, GAT, GraphSAGE, and a Hybrid ensemble — with full XAI, visualizations, and a formatted technical report.

---

## Models

| Model | Type | Classification Level |
|---|---|---|
| **E-GraphSAGE** ⭐ | Edge feature aggregation (IEEE paper) | Edge (per-flow) |
| **GCN** | Graph Convolutional Network | Node (per-device) |
| **GAT** | Graph Attention Network | Node (per-device) |
| **GraphSAGE** | Inductive neighbourhood sampling | Node (per-device) |
| **Hybrid** | GCN + GAT + GraphSAGE ensemble | Node (per-device) |

### E-GraphSAGE
Extends GraphSAGE to aggregate **edge features** (traffic flow statistics) during neighbourhood aggregation, then produces edge embeddings via `z_edge = CONCAT(z_src, z_dst)`. Each network flow is classified individually as benign or attack — which is more precise than device-level detection.

> **Reference:** Lo, W.W., et al. *E-GraphSAGE: A Graph Neural Network based Intrusion Detection System for IoT.* IEEE/IFIP NOMS 2022. [arXiv:2103.16329](https://arxiv.org/abs/2103.16329)

---

## Project Structure

```
capstone/
├── main.py                      # Entry point — run this
├── config.py                    # All default hyperparameters
├── requirements.txt
│
├── src/
│   ├── data_preprocessing.py    # Synthetic generator + BoT-IoT / ToN-IoT loaders
│   ├── graph_construction.py    # flow | knn | temporal | hybrid graph builders
│   ├── gnn_models.py            # All model definitions
│   ├── train.py                 # Trainer, evaluator, multi-model comparison
│   ├── explainability.py        # XAI: gradient saliency, attention, edge perturbation
│   ├── visualizations.py        # All plots (ROC, confusion matrix, radar, etc.)
│   └── tech_report.py           # ASCII technical report generator
│
├── data/                        # Place real dataset CSVs here
├── models/                      # Saved checkpoints + preprocessor state
├── results/                     # technical_report.txt, model_comparison.csv
└── visualizations/              # All generated plots
    ├── <ModelName>/
    │   ├── training_history.png
    │   └── confusion_matrix.png
    ├── xai/                     # Explainability plots
    ├── roc_curves.png
    ├── model_comparison.png
    ├── radar_chart.png
    ├── label_distribution.png
    └── graph_sample.png
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For PyTorch Geometric (CPU):
```bash
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
```

### 2. Run with defaults (synthetic data, all 5 models)

```bash
python main.py
```

### 3. Common overrides

```bash
# Select specific models
python main.py --models EGraphSAGE GCN GAT

# Multi-class (detect attack type instead of binary)
python main.py --multiclass

# Skip explainability step (faster)
python main.py --no-explain

# Custom JSON config
python main.py --config my_config.json
```

### 4. Real datasets

```bash
# Download BoT-IoT from https://research.unsw.edu.au/projects/bot-iot-dataset
# Place as data/bot_iot.csv
python main.py --dataset bot_iot

# ToN-IoT
python main.py --dataset ton_iot
```

---

## Configuration

All settings live in `config.py`. You can override any key via a JSON file:

```json
{
    "dataset":       "synthetic",
    "n_samples":     12000,
    "anomaly_ratio": 0.15,
    "graph_method":  "flow",
    "models":        ["EGraphSAGE", "GCN", "GAT", "GraphSAGE", "Hybrid"],
    "hidden_dims":   [128, 64],
    "dropout":       0.3,
    "heads":         4,
    "epochs":        300,
    "patience":      25,
    "learning_rate": 0.001,
    "explain":       true,
    "multiclass":    false
}
```

---

## Graph Construction

| Method | Nodes | Edges | Use with |
|---|---|---|---|
| `flow` | IP endpoints | Network flows (traffic stats) | EGraphSAGE |
| `knn` | Flow records | k-NN feature similarity | GCN, GAT, GraphSAGE, Hybrid |
| `temporal` | Flow records | Time-window adjacency | GCN, GAT, GraphSAGE, Hybrid |
| `hybrid` | IP endpoints | Flows + k-NN combined | All |

---

## Outputs

| Artifact | Path |
|---|---|
| Technical report | `results/technical_report.txt` |
| Model comparison | `results/model_comparison.csv` |
| Config snapshot | `results/config_used.json` |
| Training curves | `visualizations/<Model>/training_history.png` |
| Confusion matrices | `visualizations/<Model>/confusion_matrix.png` |
| ROC curves | `visualizations/roc_curves.png` |
| Radar chart | `visualizations/radar_chart.png` |
| XAI plots | `visualizations/xai/` |
| Model checkpoints | `models/<Model>/best_model.pth` |

---

## Explainability (XAI)

| Method | Models | Output |
|---|---|---|
| Gradient × Input Saliency | All | Which input features drive attack predictions |
| Edge Perturbation Importance | E-GraphSAGE | Which flow features most affect attack probability |
| Edge Risk Score Distribution | E-GraphSAGE | Histogram of per-flow attack probabilities |
| Attention Weight Distribution | GAT | Which edges the model attends to |

---

## Real Datasets

| Dataset | Scale | Attack Rate | Source |
|---|---|---|---|
| BoT-IoT | 3.7M flows | 99.99% | [UNSW](https://research.unsw.edu.au/projects/bot-iot-dataset) |
| ToN-IoT | 22M flows | 96.4% | [UNSW](https://research.unsw.edu.au/projects/ton-iot-datasets) |
| NF-BoT-IoT | 600K flows | 97.7% | [UQ](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) |

---

## License

Academic use only.
