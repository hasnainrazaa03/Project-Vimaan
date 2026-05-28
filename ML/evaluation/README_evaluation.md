# Model Evaluation System

Comprehensive evaluation framework for tracking model performance across versions.

---

## 📁 Folder Structure

```
ML/evaluation/
├── evaluator.py           # Core evaluation class (single model)
├── batch_evaluator.py     # Batch evaluation (all versions)
├── run_single.py          # Run single model evaluation
├── run_all.py             # Run all versions evaluation
├── results/               # Output folder (auto-created)
│   ├── metrics_v9_*.json
│   ├── comparison_all_versions_*.json
│   └── ...
└── README.md              # This file
```

---

## 🚀 Quick Start

### Evaluate Latest Model
```bash
cd ML/evaluation
python run_single.py
```

### Evaluate Specific Version
```bash
python run_single.py 9    # Evaluates v9
python run_single.py 5    # Evaluates v5
```

### Evaluate ALL Versions (v1-v9+)
```bash
python run_all.py
```

---

## 📊 What Gets Evaluated

### 1. **Overall Metrics**
- Accuracy (% correct predictions)
- Weighted F1-Score (accounts for class imbalance)
- Macro F1-Score (unweighted average)
- Cohen's Kappa (agreement score)

### 2. **Per-Intent Metrics**
- Precision (of predicted X, how many were actually X?)
- Recall (of actual X, how many did we find?)
- F1-Score (harmonic mean of precision/recall)
- Support (number of examples)

### 3. **Confidence Statistics**
- Mean confidence
- Std deviation
- Min/Max/Median

### 4. **Inference Speed**
- Mean inference time (ms)
- 95th percentile
- Min/Max times

### 5. **Confusion Matrix**
- Shows which intents are confused with each other

---

## 📈 Output Files

### Individual Metrics (per version)
```
results/metrics_v9_20251029_012345.json
```
Contains all metrics for that specific version.

### Comparison Report (all versions)
```
results/comparison_all_versions_20251029_012345.json
```
Side-by-side comparison of all versions.

---

## 📝 Example Output

### Single Model
```
==================================================
MODEL v9 - EVALUATION SUMMARY
==================================================

OVERALL METRICS
--------------------------------------------------
  Accuracy:           95.20%
  Weighted F1-Score:  0.9331
  Macro F1-Score:     0.8876
  Cohen's Kappa:      0.9234

DATASET STATISTICS
--------------------------------------------------
  Total Examples:     10,489
  Correct:            9,982 (95.2%)
  Incorrect:          507 (4.8%)
  Unique Intents:     17

...
```

### All Versions Comparison
```
======================================================================
CROSS-VERSION COMPARISON TABLE
======================================================================

Version    Accuracy     Weighted F1   Macro F1     Kappa
v1         78.54%       0.7821        0.6543       0.7234
v2         82.31%       0.8102        0.7123       0.7654
v3         85.67%       0.8456        0.7789       0.8123
...
v9         95.20%       0.9331        0.8876       0.9234

PROGRESS VISUALIZATION
==================================

ACCURACY PROGRESSION:
--------------------------------------------------
v1: ████████████████████████░░░░░░░░░░░░░░░░ 78.54%
v2: ██████████████████████████░░░░░░░░░░░░░░ 82.31%
...
v9: █████████████████████████████████████░░░ 95.20%

MODEL RANKINGS
--------------------------------------------------
🏆 Best Accuracy: v9 (95.20%)
🏆 Best Weighted F1: v9 (0.9331)
⚡ Fastest Inference: v5 (115.23ms)
```

---

## 🎯 How It Works

### evaluator.py
- Loads a single model
- Splits dataset (70% train, 15% val, 15% test)
- Runs predictions on test set
- Calculates 9 types of metrics
- Saves results to JSON

### batch_evaluator.py
- Finds all model versions (v1, v2, ..., v9, ...)
- Runs `evaluator.py` on each version
- Generates comparison table
- Creates progress visualizations
- Saves consolidated comparison JSON

### run_single.py / run_all.py
- Simple runners for the evaluators
- Handle dataset loading
- Command-line interface

---

## 🔧 Integration with Other Scripts

### From Training Script
```python
from evaluation.evaluator import ModelEvaluator

# After training
evaluator = ModelEvaluator(model_path="models/vimaan_nlu_model_best/v10")
evaluator.load_dataset("datasets/...")
metrics, _, _, _ = evaluator.evaluate_dataset(evaluator.test_data)
evaluator.save_metrics(metrics)
```

### From Analysis Script
```python
import json

# Load metrics for comparison
with open("evaluation/results/comparison_all_versions_*.json") as f:
    data = json.load(f)

# Get v9 accuracy
v9_accuracy = data['versions']['v9']['accuracy']
```

---

## 📌 Best Practices

1. **Always run evaluation after training a new model**
2. **Compare with previous versions to track progress**
3. **Use the same test set across versions** (fair comparison)
4. **Check per-intent metrics** to identify weak points
5. **Monitor inference time** for production readiness

---

## 🎯 Next Steps

After running evaluation:
1. Review metrics → Identify weak intents
2. Analyze confusion matrix → See common errors
3. Check confidence → Ensure model is calibrated
4. Compare versions → Track improvement trajectory
5. Plan retraining → Focus on weak areas
