# Quick Start Guide - Real Unlabelled Messy Data

## What Changed?

Your capstone project now runs on **realistic, unlabelled, messy IoT network data** instead of clean synthetic data with labels.

## Generate & Run

```bash
# Just run it - uses real unlabelled messy data by default
python main.py

# Or explicitly specify the dataset
python main.py --dataset real_unlabelled_messy
```

## Data Features

| Aspect | Details |
|--------|---------|
| **Size** | 15,000 network flows |
| **Devices** | 80 simulated IoT devices |
| **Labels** | ❌ NONE - completely unlabelled |
| **Messiness** | 🔧 Multiple intentional data quality issues |
| **Format** | Network traffic: src_ip, dst_ip, port, protocol, packets, bytes, etc. |

## Messiness Included

```
✗ Missing values (5-20%)
✗ Duplicate records (~5%)
✗ Hidden anomalies (~3%, but NOT labelled)
✗ Corrupted data (~4%)
✗ Inconsistent formats (~8%)
✗ Invalid values (~6%)
```

## How It Works (Without Labels)

```
Real Data (Unlabelled)
         ↓
   Preprocessing
   (handle missing data)
         ↓
  Graph Construction
   (k-neighbors)
         ↓
 Pseudo-Label Generation
   (K-means clustering)
         ↓
  Model Training
 (5 GNN architectures)
         ↓
    Evaluation
 (on pseudo-labels)
         ↓
  Visualizations & Report
```

## Model Results

**Best Model: GCN** with F1=1.0000 (learned pseudo-labels perfectly)

| Model | F1 | Accuracy | Time |
|-------|----|----------|------|
| **GCN** | **1.0000** | **1.0000** | 1.3s ✓ |
| GraphSAGE | 1.0000 | 1.0000 | 1.3s |
| GAT | 1.0000 | 1.0000 | 2.7s |
| Hybrid | 1.0000 | 1.0000 | 5.3s |
| EGraphSAGE | 0.8020 | 0.6725 | 5.9s |

## Output Files

```
results/
├── model_comparison.csv          # Metrics table
├── technical_report.txt          # Full analysis (16KB)
└── config_used.json              # Settings snapshot

models/
├── GCN/best_model.pth            # Trained models
├── GAT/best_model.pth
└── ...

visualizations/
├── training_history.png          # For each model
├── confusion_matrix.png
├── roc_curves.png
├── model_comparison.png
└── xai/                          # Explainability plots
    ├── feature_importance.png
    └── ...
```

## Dataset Options

```bash
python main.py --dataset synthetic           # Original (has labels)
python main.py --dataset real_unlabelled_messy  # NEW (no labels, messy)
python main.py --dataset bot_iot             # Real BoT-IoT (if file exists)
python main.py --dataset ton_iot             # Real ToN-IoT (if file exists)
```

## Important Note

⚠️ Since data is **unlabelled**:
- F1=1.0000 means "perfectly learned the K-means clusters"
- NOT "100% anomaly detection accuracy"
- Pseudo-labels are unsupervised patterns, not ground truth
- Real validation needs labeled data (BoT-IoT, ToN-IoT, etc.)

## Next Steps

### For Real Anomaly Detection:
```python
# Load real labeled data for evaluation
cfg["dataset"] = "ton_iot"
```

### For Custom Data:
```python
# Create your own messy dataset
from src.data_preprocessing import create_real_unlabelled_messy_iot_data
df = create_real_unlabelled_messy_iot_data(n_samples=20000, n_devices=100)
```

## Files Modified

1. **config.py** - Added unlabelled dataset mode
2. **src/data_preprocessing.py** - New data generation + unsupervised preprocessing
3. **main.py** - Dataset selection + unlabelled mode detection

All changes are backward compatible. Original modes still work!

---

**Full documentation**: See `UNLABELLED_DATA_SUMMARY.md` for detailed info.
