# UK Credit Default — Imbalanced Classification Pipeline

A reproducible machine learning project for **credit default prediction**, focused on:

- Imbalanced classification
- Model selection via cross-validation
- Proper evaluation using ROC AUC and PR AUC
- Threshold optimisation
- Comparison of Logistic Regression, Random Forest and XGBoost

---

## Problem

Predict whether a customer will default (binary classification).

The dataset is imbalanced:
- ~22% default rate
- ~78% non-default

This makes accuracy misleading and requires careful evaluation.

---

## Baseline Model (Vanilla Approach)

The initial version:

- Used standard classifiers
- Selected using default metrics
- Applied default probability threshold = 0.5

### Baseline behaviour

- High accuracy (dominated by majority class)
- Low recall on defaulters (~35%)
- Many false negatives (missed defaulters)

This reflected a common issue in imbalanced datasets:
> Models optimise accuracy and under-detect the minority class.

---

## Improved Pipeline

The updated version introduces:

### 1. Imbalance-aware model selection
- Cross-validation scoring using **average precision (PR AUC)** instead of accuracy
- PR AUC is more informative when positive class is rare

### 2. Class weighting
- Logistic Regression: `class_weight="balanced"`
- Random Forest: `class_weight="balanced_subsample"`
- XGBoost: `scale_pos_weight = (#neg / #pos)`

This increases sensitivity to defaulters.

### 3. Threshold optimisation
Instead of using 0.5 blindly, the decision threshold is selected via grid search to optimise:

- F1 score (default)
- or recall (optionally subject to minimum precision)

---

## Final Model Performance

**Best model:** Random Forest  
**Best CV metric (average precision):** 0.559  

### Test Metrics

- ROC AUC: **0.77**
- PR AUC: **0.55**
- Recall (defaults): **~59%**
- Precision (defaults): **~50%**
- F1 score: **~0.54**

### Confusion Matrix (Optimised Threshold ≈ 0.493)

|                | Pred 0 | Pred 1 |
|----------------|--------|--------|
| True 0         | 3881   | 792    |
| True 1         | 544    | 783    |

Compared to the baseline model:

- Recall improved from ~35% → ~59%
- The model now captures substantially more defaulters
- Precision decreases slightly (expected tradeoff)
- Overall F1 improved

This demonstrates the importance of:
- Proper metric choice
- Class weighting
- Threshold tuning

---

## Interpretation

- ROC AUC 0.77 indicates good ranking ability.
- PR AUC 0.55 is ~2.5× better than random baseline (~0.22).
- The model meaningfully improves detection of defaulters without extreme overprediction.

The tradeoff between false positives and false negatives can be adjusted via threshold tuning.

---

## Threshold Analysis

The project includes:

- ROC curve
- Precision–Recall curve
- Threshold sweep (precision/recall/F1 vs threshold)

This allows explicit control over:

- Conservative policy (high precision)
- Aggressive risk detection (high recall)
- Balanced F1 optimisation

---

## Possible Extensions

- Cost-sensitive threshold optimisation  
  (minimise expected loss instead of maximising F1)

- Probability calibration (Platt scaling / isotonic regression)

- Feature importance and SHAP values

- Time-series extension with rolling validation

---

## Quickstart

### Install
```bash
pip install -r requirements.txt
```

### Download dataset
```bash
python download_data.py
```

### Train and evaluate
```bash
python run_all.py --metric average_precision
```

Outputs:
- `results/metrics.json`
- `results/plots/confusion_matrix.png`
- `results/plots/roc_curve.png`
- `results/plots/pr_curve.png`
- `results/plots/threshold_sweep.png`

---

## Key Takeaways

This project illustrates how naive modelling choices can underperform in imbalanced problems, and how:

- metric selection,
- class weighting,
- and threshold optimisation

materially improve model performance.

It reflects a progression from a baseline classifier to a more structured imbalanced classification pipeline.
