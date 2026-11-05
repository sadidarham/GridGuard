# GridGuard: Power Grid Stability Prediction Using Machine Learning

Course project for CS 746 Perspectives on Data Science (Wichita State University, Fall 2026).
Author: Arham Sadid Hossain

## Overview

GridGuard uses supervised machine learning to classify a **simulated** decentralized electrical-grid condition as **stable** or **unstable**, based on participant reaction times, power production/consumption, and price-response values.

This is a course prototype. It uses simulated data from a single four-node grid and does **not** control real equipment or represent a real utility system. It is not production-ready and makes no claim about preventing real-world outages.

**Research question:** How accurately can machine-learning models predict whether a decentralized power-grid condition is stable or unstable using reaction times, power, and price-response data?

## Dataset

- **Name:** Electrical Grid Stability Simulated Data
- **Source:** UCI Machine Learning Repository, dataset ID 471 (DOI 10.24432/C5PG66)
- **Creator:** Vadim Arzamasov (2018)
- **Link:** https://archive.ics.uci.edu/dataset/471/electrical+grid+stability+simulated+data
- **Size:** 10,000 rows, 14 columns, no missing values, no duplicate rows

Each row is one simulated operating condition of a four-node star grid (one producer, three consumers).

| Columns | Meaning |
|---|---|
| `tau1`-`tau4` | Reaction time of the producer (1) and consumers (2-4) |
| `p1`-`p4` | Nominal power (producer positive, consumers negative) |
| `g1`-`g4` | Price elasticity of each participant |
| `stab` | Continuous stability value (**not used as an input**) |
| `stabf` | Label: `stable` or `unstable` (prediction target) |

Class balance: 6,380 unstable (63.8%) and 3,620 stable (36.2%).

## Key Methodological Decisions

- **`stab` removed (target leakage).** `stabf` is derived directly from `stab` (positive = unstable, negative = stable). Including it would hand the model the answer.
- **`p1` removed (redundant).** `p1 = abs(p2 + p3 + p4)`; verified across all rows (max difference about 6e-15, floating-point noise).
- **Stratified 80/20 train/test split** (`random_state=42`), so both sets keep the 63.8% / 36.2% class ratio.
- **Scaling inside a Pipeline** for Logistic Regression, so the scaler is fit on training data only.
- **Positive class = unstable** (coded as 1), so precision and recall refer to the unstable class. A false negative (unstable predicted as stable) is treated as the more serious error.

Final input features (11): `tau1`-`tau4`, `p2`-`p4`, `g1`-`g4`.

## Results

All numbers are on the **held-out test set (2,000 rows)**, a single split with seed 42, with no hyperparameter tuning.

| Model | Accuracy | Precision (unstable) | Recall (unstable) | F1 (unstable) | Missed unstable (FN) |
|---|---|---|---|---|---|
| Logistic Regression | 0.820 | 0.841 | 0.886 | 0.863 | 146 |
| Decision Tree (max_depth=5, min_samples_leaf=20) | 0.792 | 0.822 | 0.861 | 0.841 | 178 |
| Random Forest (200 trees, min_samples_leaf=5) | 0.928 | 0.928 | 0.962 | 0.945 | 48 |

Baseline for context: always predicting "unstable" gives 63.8% accuracy.

**Observations**
- Reaction times (`tau1`-`tau4`) and price elasticities (`g1`-`g4`) show the strongest relationship with instability in the EDA, logistic coefficients, and Random Forest feature importance. Consumer power (`p2`-`p4`) contributes little.
- The Random Forest outperformed the linear model and shallow tree, which suggests interactions or nonlinear effects. This is an interpretation, not a proven cause.
- The Random Forest fits training data much better (train accuracy about 0.99) than test data (0.928), so test metrics are the ones to trust.
- Feature importance describes what the model used, not what causes instability.

Figures are in `reports/figures/`: class counts, box plots by stability, correlation heatmap, feature importance, and model comparison.

## Project Structure

```
GridGuard/
├── data/
│   ├── raw/            # original UCI file, never edited (not tracked by git)
│   └── processed/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training.ipynb
├── reports/figures/
├── requirements.txt
└── README.md
```

## How to Run

```bash
git clone https://github.com/sadidarham/GridGuard.git
cd GridGuard
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# download the raw data from UCI
mkdir -p data/raw
curl -L -o data/raw/grid_stability.zip "https://archive.ics.uci.edu/static/public/471/electrical+grid+stability+simulated+data.zip"
unzip data/raw/grid_stability.zip -d data/raw
```

Then open the notebooks in order (`01_data_exploration.ipynb`, then `02_model_training.ipynb`) with the `.venv` kernel. The notebooks expect `data/raw/Data_for_UCI_named.csv`.

## Limitations

- The data is **simulated** and generated from fixed mathematical assumptions; features are sampled independently (near-zero correlation between them), unlike real grid telemetry.
- It covers **one four-node star grid**, so results will not automatically generalize to other grid structures or real grids.
- Results come from **one train/test split and one random seed**; no cross-validation or hyperparameter tuning yet, so reported numbers could shift.
- Models are not calibrated for real operational use, and this project does not include voltage, frequency, weather, demand, or time-series data.

## Status

Completed: data acquisition, inspection, EDA, leakage-safe preprocessing, three baseline models with test-set evaluation, feature importance.

Still to do: confusion-matrix figures, optional cross-validation/tuning, `src/` scripts, final report.

## Tools

Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, Jupyter, Git/GitHub.
