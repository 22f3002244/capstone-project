# IoT Anomaly Detection using Graph Neural Networks (GNN-based NIDS)

A comprehensive, production-quality framework for detecting network intrusions and anomalies in IoT environments using **Graph Neural Networks**. This project implements state-of-the-art GNN architectures including **E-GraphSAGE** (Lo et al., IEEE/IFIP NOMS 2022) as the flagship model, alongside GCN, GAT, GraphSAGE, and a Hybrid ensemble.

## Overview

**Problem:** IoT networks face increasing security threats with limited computational resources on edge devices. Traditional anomaly detection methods struggle with the heterogeneous, dynamic nature of IoT traffic.

**Solution:** This project leverages Graph Neural Networks to model IoT network topology as a graph where:
- **Nodes** = IP devices/endpoints
- **Edges** = Network flows (traffic connections) with rich feature vectors
- **Classification** = Detecting malicious flows/devices using learned node/edge embeddings

**Key Features:**
- ✅ **Multi-model comparison**: 5 state-of-the-art GNN architectures
- ✅ **E-GraphSAGE**: Attack flow (edge) detection, not just device classification
- ✅ **Real & synthetic data**: Works with BoT-IoT, ToN-IoT, or custom IoT datasets
- ✅ **Unlabeled learning**: K-means pseudo-labeling for unsupervised scenarios
- ✅ **Explainability**: XAI methods (gradient saliency, attention, edge perturbation)
- ✅ **Complete pipeline**: Data → Preprocessing → Graph → Training → Evaluation → Report
- ✅ **Production outputs**: Checkpoints, visualizations, technical report

---

## Models & Architectures

### Model Comparison Table

| Model | Type | Scope | Parameters | Key Advantage |
|---|---|---|---|---|
| **E-GraphSAGE** ⭐ | Edge Aggregation (IEEE paper) | Edge (per-flow) | 21,314 | Detects individual malicious flows, not just devices |
| **GCN** | Graph Convolutional | Node (per-device) | 10,562 | Lightweight, interpretable, fastest training |
| **GAT** | Graph Attention | Node (per-device) | 140,550 | learns which neighbors matter via attention |
| **GraphSAGE** | Sample & Aggregate | Node (per-device) | 20,418 | Inductive, scalable to new nodes |
| **Hybrid** | GCN + GAT + GraphSAGE Ensemble | Node (per-device) | 83,970 | Combines strengths, better generalization |

### Model Details

#### E-GraphSAGE (Flagship Model)
**Paper:** Lo, W.W., et al. *E-GraphSAGE: A Graph Neural Network based Intrusion Detection System for IoT.* IEEE/IFIP NOMS 2022. ([arXiv:2103.16329](https://arxiv.org/abs/2103.16329))

**Innovation:** Aggregates edge features (flow statistics) during neighborhood aggregation:
1. Each node (device) has its neighbors' edge features
2. Aggregate these edge features to create flow-aware embeddings
3. Classify at **edge level**: $z_{edge} = \text{CONCAT}(z_{src}, z_{dst})$
4. Output: Per-flow attack/benign prediction (not just per-device)

**Advantage:** Detects specific attack flows from otherwise benign devices — critical for IoT where devices may send mixed traffic.

#### GCN - Graph Convolutional Networks
- Simple but powerful: applies convolution to graph neighborhoods
- Fastest training (1.5-2.5s on 15K samples)
- Good for: resource-constrained environments, edge deployment
- Limitation: assumes all neighbors equally important (no attention)

#### GAT - Graph Attention Network
- Learns *which* neighbors matter via multi-head attention
- Excellent for identifying important connections in IoT networks
- Good for: understanding traffic patterns, interpretability
- Limitation: higher memory usage on large graphs

#### GraphSAGE - Inductive Learning
- "Sample and aggregate" approach: inductive (works on unseen nodes)
- Efficient on large networks via sampling
- Good for: dynamic IoT networks where new devices appear
- Limitation: requires neighborhood sampling configuration

#### Hybrid - Ensemble
- Combines GCN + GAT + GraphSAGE predictions
- Better generalization via ensemble voting
- Good for: critical deployments where confidence matters
- Limitation: slowest training (combines 3 models)

---

## Project Structure & Architecture

```
capstone/
├── main.py                          # 🎯 Entry point — orchestrates entire pipeline
├── config.py                        # ⚙️ All hyperparameters (easily overridable)
├── requirements.txt                 # 📦 Python dependencies
├── README.md                        # 📖 This file
│
├── src/                             # 🔧 Core implementation modules
│   ├── __init__.py
│   ├── data_preprocessing.py        # Data generation, loading, feature engineering
│   │   ├── create_real_unlabelled_messy_iot_data()   # Synthetic messy IoT data
│   │   ├── create_synthetic_iot_data()               # Baseline synthetic generator
│   │   ├── load_bot_iot()                            # Real BoT-IoT dataset
│   │   ├── load_ton_iot()                            # Real ToN-IoT dataset
│   │   └── IoTDataPreprocessor                       # Scaling, encoding, pseudo-labels
│   │
│   ├── graph_construction.py        # Build graph from network flows
│   │   ├── FlowGraphBuilder              # IP-centric: devices as nodes, flows as edges
│   │   ├── KNNGraphBuilder               # k-NN: connect similar flow vectors
│   │   ├── TemporalGraphBuilder          # Time-window: flows within time windows
│   │   └── HybridGraphBuilder            # Combination of multiple methods
│   │
│   ├── gnn_models.py                # Neural network architectures
│   │   ├── EGraphSAGE               # Edge feature aggregation
│   │   ├── GCN                      # Graph Convolutional Network
│   │   ├── GAT                      # Graph Attention Network
│   │   ├── GraphSAGE                # Inductive neighborhood sampling
│   │   └── HybridGNN                # Ensemble predictor
│   │
│   ├── train.py                     # Training loop & evaluation
│   │   ├── GNNTrainer               # Single model trainer
│   │   ├── evaluate_model()         # Compute metrics (F1, AUC, accuracy, etc.)
│   │   └── compare_models()         # Multi-model comparison table
│   │
│   ├── explainability.py            # XAI methods for interpretability
│   │   ├── GradientSaliency         # Gradient × Input saliency
│   │   ├── EdgePerturbation         # Which flow features matter (E-GraphSAGE)
│   │   ├── AttentionAnalysis        # GAT attention distribution
│   │   └── ExplainabilityAnalyzer   # Unified interface
│   │
│   ├── visualizations.py            # Plot generation
│   │   ├── plot_training_history()  # Loss/accuracy curves
│   │   ├── plot_confusion_matrix()  # Confusion matrix heatmap
│   │   ├── plot_roc_curves()        # ROC-AUC for all models
│   │   ├── plot_model_comparison()  # Bar charts of metrics
│   │   ├── plot_radar_chart()       # Radar chart of all metrics
│   │   ├── plot_graph_sample()      # Visualize graph structure
│   │   └── plot_label_distribution()# Class balance visualization
│   │
│   └── tech_report.py               # Technical report generator
│       └── generate_report()        # Formatted ASCII report with statistics
│
├── data/                            # 📊 Dataset directory (user-provided)
│   ├── bot_iot.csv                  # (if using --dataset bot_iot)
│   ├── ton_iot.csv                  # (if using --dataset ton_iot)
│   └── nf_bot_iot.csv               # (if using --dataset nf_bot_iot)
│
├── models/                          # 💾 Saved checkpoints & preprocessor
│   ├── preprocessor.pkl             # Fitted scaler, encoder, K-means
│   └── <ModelName>/
│       └── best_model.pth           # PyTorch checkpoint for each model
│
├── results/                         # 📈 Final outputs
│   ├── technical_report.txt         # Formatted report with all metrics/recommendations
│   ├── model_comparison.csv         # CSV table of model metrics
│   └── config_used.json             # Config snapshot for reproducibility
│
└── visualizations/                  # 🖼️ All generated plots
    ├── <ModelName>/                 # Per-model visualizations
    │   ├── training_history.png
    │   ├── confusion_matrix.png
    │   ├── roc_curve.png
    │   └── feature_importance.png
    ├── xai/                         # Explainability visualizations
    │   ├── EGraphSAGE_edge_risk.png
    │   ├── EGraphSAGE_gradient_saliency.png
    │   ├── GAT_attention_distribution.png
    │   └── ...
    ├── roc_curves.png               # All models' ROC curves overlaid
    ├── model_comparison.png         # Metrics comparison bar charts
    ├── radar_chart.png              # Multi-metric radar chart
    ├── graph_sample.png             # Visualization of IoT network graph
    └── label_distribution.png       # Class balance visualization
```

### File Responsibilities

| File | Purpose | Key Functions |
|---|---|---|
| `main.py` | Pipeline orchestration | Loads → Preprocesses → Trains → Evaluates → Reports |
| `config.py` | Configuration management | Default hyperparameters |
| `data_preprocessing.py` | Data handling | Generate/load, scale, encode, pseudo-label |
| `graph_construction.py` | Graph building | IP-centric, k-NN, temporal, hybrid methods |
| `gnn_models.py` | Neural networks | GNN architectures |
| `train.py` | Model training | Trainer, loss functions, evaluation |
| `explainability.py` | Interpretability | Gradient saliency, attention, edge perturbation |
| `visualizations.py` | Plotting | ROC, confusion matrix, training curves, etc. |
| `tech_report.py` | Report generation | Formatted ASCII report with statistics |

---

## Installation & Setup

### Prerequisites
- **Python 3.9+** (tested on 3.11)
- **CUDA 11.8+** (optional, for GPU acceleration)
- **4GB+ RAM** (8GB+ recommended for large datasets)

### 1. Clone & Navigate

```bash
git clone <repo-url>
cd capstone
```

### 2. Create Virtual Environment (Recommended)

```bash
# Python venv
python -m venv venv
source venv/bin/activate      # Linux/Mac
# or
venv\Scripts\activate          # Windows

# Or Conda
conda create -n iot-nids python=3.11
conda activate iot-nids
```

### 3. Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# For PyTorch Geometric (CPU version)
pip install torch-geometric

# For GPU support (replace CPU with cu118 for CUDA 11.8)
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

### 4. Verify Installation

```bash
python -c "import torch; import torch_geometric; print('✅ All dependencies installed')"
```

---

## Quick Start

### Basic Usage (Synthetic Data)

```bash
# Run full pipeline with default settings
python main.py
```

**What happens:**
1. Generates 15,750 synthetic IoT flows (unlabeled, messy)
2. Preprocesses: cleaning, normalization, K-means pseudo-labeling
3. Builds flow graph: 114 IP devices, 15,750 edges (flows)
4. Trains 5 GNN models in parallel
5. Evaluates each model on test set
6. Generates plots, XAI analysis, technical report

**Output:** Check `results/` and `visualizations/`

### Advanced Examples

#### Use Real Dataset (BoT-IoT)

```bash
# Download from https://research.unsw.edu.au/projects/bot-iot-dataset
# Place as data/bot_iot.csv

python main.py --dataset bot_iot
```

#### Select Specific Models

```bash
# Train only GCN and E-GraphSAGE (faster)
python main.py --models EGraphSAGE GCN
```

#### Multi-Class Classification

```bash
# Detect attack type instead of binary (normal vs attack)
python main.py --multiclass
```

#### Custom Configuration

```bash
# Create my_config.json
cat > my_config.json <<EOF
{
    "dataset": "synthetic",
    "n_samples": 20000,
    "anomaly_ratio": 0.2,
    "graph_method": "knn",
    "models": ["EGraphSAGE", "GAT"],
    "hidden_dims": [256, 128],
    "epochs": 500,
    "learning_rate": 0.0005,
    "explain": true
}
EOF

python main.py --config my_config.json
```

#### Skip XAI (Faster Training)

```bash
# Explainability is slow; skip it for quick runs
python main.py --no-explain
```

#### No Visualization (Fastest)

```bash
python main.py --no-viz --no-explain
```

---

## Configuration Guide

### Configuration File (config.py)

All settings are centralized in `config.py` and can be overridden via:
1. Command-line arguments: `python main.py --param value`
2. JSON file: `python main.py --config config.json`
3. Direct editing of `config.py`

### Key Parameters Explained

```json
{
    "dataset": "real_unlabelled_messy",          # Data source
                                                  # Options: "synthetic", "bot_iot", "ton_iot", 
                                                  #          "nf_bot_iot", "real_unlabelled_messy"
    
    "n_samples": 15000,                           # Number of IoT flows to use/generate
    
    "anomaly_ratio": 0.15,                        # % of attack flows (for synthetic only)
    
    "n_devices": 80,                              # Number of IoT devices in network
    
    "unlabelled": true,                           # True = no labels, use K-means pseudo-labels
                                                  # False = use dataset labels
    
    "graph_method": "flow",                       # How to build graph
                                                  # "flow" = IP devices as nodes, flows as edges
                                                  # "knn" = flows as nodes, k-NN edges
                                                  # "temporal" = time-window based
                                                  # "hybrid" = combination
    
    "k_neighbors": 7,                             # For k-NN graph: # neighbors per node
    
    "models": ["EGraphSAGE", "GCN", "GAT",       # Which models to train
               "GraphSAGE", "Hybrid"],
    
    "hidden_dims": [128, 64],                    # Hidden layer sizes for GNNs
                                                  # Each number = one GNN layer
    
    "dropout": 0.3,                              # Dropout rate to prevent overfitting
    
    "heads": 4,                                   # Multi-head attention (GAT only)
    
    "epochs": 300,                               # Max training epochs
    
    "patience": 25,                              # Early stopping: stop if no improvement
                                                  # for this many epochs
    
    "batch_size": 32,                            # Batch size for gradient descent
    
    "learning_rate": 0.001,                      # Optimizer step size
    
    "weight_decay": 0.0005,                      # L2 regularization strength
    
    "train_ratio": 0.7,                          # 70% train, 15% val, 15% test
    
    "explain": true,                             # Generate XAI visualizations
    
    "multiclass": false                          # Binary = normal vs attack
                                                  # Multiclass = attack type classification
}
```

### Hyperparameter Tuning Tips

| Parameter | Impact | Tuning Strategy |
|---|---|---|
| `hidden_dims` | Model capacity | Start [64, 32], increase if underfitting |
| `dropout` | Overfitting | 0.3-0.5 typical; increase if overfitting |
| `learning_rate` | Convergence | 0.001-0.01; smaller = slower but more stable |
| `epochs` | Training time | Higher = better fit but slower; use patience to stop early |
| `patience` | Early stopping | 10-30; larger = more patient, may overfit |
| `batch_size` | Memory/speed | 32-128; larger = faster but requires more RAM |

---

## Datasets

### Synthetic Data (Default)

**Real Unlabeled Messy Data:**
```bash
python main.py --dataset real_unlabelled_messy
```

Generated with realistic messiness:
- **15,750 flows** across 80 IoT devices
- **Missing values**: 5-33% per feature (power outages, sensor failures)
- **Duplicates**: ~5% (repeated traffic patterns)
- **Outliers**: ~3% hidden anomalies
- **Inconsistent values**: ~8% (encoding errors)
- **No labels** → K-means pseudo-labeling (1% vs 99% imbalance)

**Standard Synthetic:**
```bash
python main.py --dataset synthetic
```

- Programmatically generated IoT traffic
- Configurable anomaly ratio, device count
- Clean (no missing values)
- Labeled ground truth

### Real Datasets

#### BoT-IoT (Botnet-IoT)
```bash
python main.py --dataset bot_iot
```

**Source:** UNSW Sydney - https://research.unsw.edu.au/projects/bot-iot-dataset
- **3.7 million** flows
- **99.99%** attack rate (heavily imbalanced)
- **11 attack types**: DDoS, DoS, SSH, HTTP port-scan, Telnet, Mirai variants
- **45 features**: TCP/UDP stats, packet sizes, durations, IAT
- Download: See dataset page for access

#### ToN-IoT (Things-on-Network-IoT)
```bash
python main.py --dataset ton_iot
```

**Source:** UNSW Sydney - https://research.unsw.edu.au/projects/ton-iot-datasets
- **22 million** flows
- **96.4%** attack rate
- **6 attack types**: DDoS, DoS, Backdoor, Injection, Man-in-Middle, Reconnaissance
- **51 features**: More detailed flow analysis
- Across normal & compromised IoT devices

#### NF-BoT-IoT (NetFlow BoT-IoT)
```bash
python main.py --dataset nf_bot_iot
```

**Source:** UQ Labs - https://staff.itee.uq.edu.au/marius/NIDS_datasets/
- **600K** NetFlow v5 records
- **97.7%** attack rate
- **10 features**: NetFlow format (duration, bytes, packets, flags, etc.)
- Lightweight, good for resource-constrained scenarios

### Using Custom Datasets

1. Prepare CSV with format:
```csv
src_ip,dst_ip,protocol,src_port,dst_port,packet_count,byte_count,...,label
192.168.1.1,192.168.1.2,TCP,50000,443,45,4500,...,0
...
```

2. Place in `data/` directory

3. Update `data_preprocessing.py`:
```python
elif dataset_name == "custom":
    return load_custom_dataset("data/custom.csv")
```

4. Run: `python main.py --dataset custom`

---

## Graph Construction Methods

The project supports multiple ways to convert IoT flows into a graph structure:

### 1. Flow Graph (Default for E-GraphSAGE)

```
Nodes = IP devices/endpoints
Edges = Network flows (traffic connections)
Edge Features = 13D: packet_count, byte_count, duration, inter_arrival_time, etc.
```

**Use case:** Device-to-device communication patterns
**Benefits:** Intuitive, interpretable, IP-centric view
**Best for:** E-GraphSAGE (edge-level classification)

### 2. k-NN Graph

```
Nodes = Individual flows
Edges = k-nearest neighbors (7 by default)
Weight = Feature similarity (Euclidean distance)
```

**Use case:** Grouping similar flows together
**Benefits:** Discovers attack patterns via similarity
**Best for:** GCN, GAT, GraphSAGE, Hybrid

### 3. Temporal Graph

```
Nodes = Flows
Edges = Flows within sliding time window
Direction = Time-ordered sequence
```

**Use case:** Sequential/multi-stage attacks
**Benefits:** Captures causal dependencies over time
**Best for:** Slow attacks, APT detection

### 4. Hybrid Graph

Combines strengths of Flow + k-NN + Temporal methods

---

## Outputs & Results

### 1. Technical Report

**Location:** `results/technical_report.txt`

Contains:
- Experiment configuration
- Per-model performance (metrics, confusion matrix, timing)
- Comparative analysis and statistical summary
- Key findings and model recommendations
- Deployment guidance and retraining schedule

### 2. Model Comparison CSV

**Location:** `results/model_comparison.csv`

Easy to import into Excel/Pandas for analysis

### 3. Visualizations

All plots saved to `visualizations/`:

```
visualizations/
├── EGraphSAGE/          # Per-model plots
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── GCN/
├── GAT/
├── GraphSAGE/
├── Hybrid/
├── xai/                 # Explainability visualizations
│   ├── *_gradient_saliency.png
│   ├── *_edge_risk.png
│   ├── *_feature_importance.png
│   └── *_attention.png
├── roc_curves.png       # All models overlaid
├── model_comparison.png # Bar charts
├── radar_chart.png      # Multi-metric radar
├── graph_sample.png     # Network topology
└── label_distribution.png
```

### 4. Checkpoints

**Location:** `models/`

Saved PyTorch models for each trained architecture. Load with:

```python
import torch
from src.gnn_models import GCN

model = GCN(input_dim=13, hidden_dims=[128, 64], num_classes=2)
state_dict = torch.load('models/GCN/best_model.pth')
model.load_state_dict(state_dict)
model.eval()
```

---

## Explainability (XAI) Methods

### Gradient-Based Saliency

**Method:** $\text{Saliency} = \text{Input} \times \nabla_{\text{Input}} \text{Prediction}$

**Interpretation:** Which features push the model towards attack prediction?

**Visualization:** Heatmap of feature importance scores

### Edge Perturbation (E-GraphSAGE Only)

**Method:** Remove edges one-by-one and measure prediction impact

**Interpretation:** Which flows are most critical for attack detection?

**Result:** Ranking of flows by importance

### Attention Weights (GAT)

**Method:** Visualize learned multi-head attention over neighbors

**Interpretation:** Which connections does the model focus on?

**Application:** Understanding which devices' traffic patterns matter

### Edge Risk Distribution

**Method:** Histogram of predicted attack probability per edge

**Interpretation:**
- Bimodal → Model confident (clean separation)
- Uniform → Model uncertain
- Skewed left → Mostly benign
- Skewed right → Mostly attack

---

## Performance Metrics Explained

| Metric | Formula | Interpretation |
|---|---|---|
| **Accuracy** | (TP + TN) / Total | % correct predictions (can be misleading on imbalanced data) |
| **Precision** | TP / (TP + FP) | Of predicted attacks, how many were correct? (minimize false alarms) |
| **Recall** | TP / (TP + FN) | Of real attacks, how many did we catch? (minimize missed attacks) |
| **F1-Score** | 2 × (Prec × Rec) / (Prec + Rec) | Harmonic mean (best for imbalanced data) |
| **AUC-ROC** | Area under ROC curve | Classifier's ability to rank; 0.5 = random, 1.0 = perfect |

**For NIDS deployment:** Prioritize **Recall** (don't miss attacks) and **Precision** (avoid false alarms)

---

## Troubleshooting

### Out of Memory (OOM)

**Error:** `RuntimeError: CUDA out of memory` or similar

**Solutions:**
```bash
# Reduce batch size
python main.py --batch_size 16

# Use CPU instead of GPU
export CUDA_VISIBLE_DEVICES=""
python main.py

# Reduce hidden dimensions
python main.py --hidden_dims 64 32

# Train fewer models
python main.py --models GCN GraphSAGE
```

### Slow Training / Timeout

**Error:** Pipeline takes > 5 minutes on synthetic data

**Solutions:**
```bash
# Skip expensive steps
python main.py --no-explain --no-viz

# Reduce epochs
python main.py --epochs 100

# Train fewer models
python main.py --models GCN GAT
```

### Dataset Not Found

**Error:** `FileNotFoundError: data/bot_iot.csv not found`

**Solutions:**
```bash
# Download dataset from official source
# Place CSV file in data/ directory

# Or use synthetic data
python main.py --dataset synthetic
```

### GPU Not Detected

**Error:** `device = cuda but GPU not available`

**Solutions:**
```bash
# Check CUDA installation
nvidia-smi

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or force CPU
export CUDA_VISIBLE_DEVICES=""
python main.py
```

### Models Training to ~100% Accuracy (Seems Too Good)

**Note:** This is expected behavior on pseudo-labeled data

On randomly generated data with K-means pseudo-labels, GNNs achieve near-perfect accuracy because:
- K-means creates geometrically separated clusters
- GNNs excel at fitting to cluster patterns
- The "ground truth" is just the clustering artifact

**For realistic evaluation:** Use real datasets (BoT-IoT, ToN-IoT) with genuine labels

---

## Deployment Guide

### Edge Device Deployment

1. **Lightweight model:** Use `GCN` (1.5-2.5s training, smallest footprint)
2. **Export as ONNX:**
   ```python
   import torch.onnx
   model = GCN(13, [64], 2)
   torch.onnx.export(model, example_input, "gcn.onnx")
   ```
3. **Run inference:** Use ONNX Runtime (C++, Java, C#, Python)
4. **Update frequency:** Retrain every 30-90 days or when F1 drops below 0.90

### Ensemble Deployment

```python
# Load all models
models = {
    'gcn': torch.load('models/GCN/best_model.pth'),
    'gat': torch.load('models/GAT/best_model.pth'),
    'graphsage': torch.load('models/GraphSAGE/best_model.pth'),
}

# Voting: predict attack if ≥2/3 models agree
predictions = torch.stack([m(data) for m in models.values()])
final = (predictions.mean(dim=0) > 0.5).long()
```

### Threshold Optimization

Default threshold = 0.5 (50% attack probability)

**Adjust based on use case:**
```python
# Conservative (fewer false alarms): threshold = 0.7
# Aggressive (fewer missed attacks): threshold = 0.3
```

---

## Citation

If you use this project, please cite:

```bibtex
@inproceedings{lo2022egraphsage,
  title={E-GraphSAGE: A Graph Neural Network based Intrusion Detection System for IoT},
  author={Lo, W.W. and Layeghy, S. and Sarhan, M. and Portmann, M.},
  booktitle={IEEE/IFIP NOMS 2022},
  year={2022},
  organization={IEEE}
}

@misc{capstone2026,
  title={IoT Anomaly Detection Framework: GNN-based NIDS},
  author={Your Name},
  year={2026},
  howpublished={\url{https://github.com/your-repo}}
}
```

---

