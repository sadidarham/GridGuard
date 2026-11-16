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

- **`stab` removed (target leakage).** `stabf` is derived directly from `stab` (positive = unstable, negative = stable). Including it would hand the model the answer. The code asserts that `stab` is never an input.
- **`p1` removed (redundant).** `p1 = abs(p2 + p3 + p4)`; verified across all rows (max difference about 6e-15, floating-point noise).
- **Stratified 80/20 train/test split** (`random_state=42`), so both sets keep the 63.8% / 36.2% class ratio.
- **Scaling inside a Pipeline** for Logistic Regression, so the scaler is fit on training data only.
- **Hyperparameters tuned on the training set only.** Grid search with 5-fold stratified cross-validation, selecting on F1. The test set was used once, after settings were fixed.
- **Positive class = unstable** (coded as 1), so precision and recall refer to the unstable class. A false negative (unstable predicted as stable) is treated as the more serious error.

Final input features (11): `tau1`-`tau4`, `p2`-`p4`, `g1`-`g4`.

## Results

Test-set results (2,000 held-out rows, unstable = positive class). One split, seed 42.

**Tuned models**

| Model | Accuracy | Precision | Recall | F1 | Missed unstable (FN) |
|---|---|---|---|---|---|
| Logistic Regression (C=0.1) | 0.820 | 0.840 | 0.886 | 0.862 | 145 |
| Decision Tree (max_depth=12, min_samples_leaf=20) | 0.852 | 0.875 | 0.897 | 0.886 | 132 |
| Random Forest (400 trees, min_samples_leaf=2) | 0.930 | 0.931 | 0.961 | 0.946 | 50 |

**Initial settings** (Logistic Regression C=1; Decision Tree max_depth=5, min_samples_leaf=20; Random Forest 200 trees, min_samples_leaf=5)

| Model | Accuracy | Precision | Recall | F1 | Missed unstable (FN) |
|---|---|---|---|---|---|
| Logistic Regression | 0.820 | 0.841 | 0.886 | 0.863 | 146 |
| Decision Tree | 0.792 | 0.822 | 0.861 | 0.841 | 178 |
| Random Forest | 0.928 | 0.928 | 0.962 | 0.945 | 48 |

**5-fold cross-validation on the training set (initial settings, mean ± std)**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.815 ± 0.007 | 0.839 ± 0.006 | 0.878 ± 0.008 | 0.858 ± 0.006 |
| Decision Tree | 0.792 ± 0.012 | 0.803 ± 0.017 | 0.893 ± 0.021 | 0.845 ± 0.008 |
| Random Forest | 0.914 ± 0.008 | 0.913 ± 0.006 | 0.957 ± 0.007 | 0.935 ± 0.006 |

Baseline for context: always predicting "unstable" gives 63.8% accuracy.

**Observations**
- Reaction times (`tau1`-`tau4`) and price elasticities (`g1`-`g4`) show the strongest relationship with instability in the EDA, logistic coefficients, and Random Forest feature importance. Consumer power (`p2`-`p4`) contributes little.
- The Random Forest is the strongest model by a wide margin, and cross-validation agrees with the test set.
- Tuning mainly helped the Decision Tree (accuracy 0.792 to 0.852); Logistic Regression and the Random Forest barely changed. The ranking of Logistic Regression vs the Decision Tree on recall is not stable across evaluations.
- The Random Forest's lead likely reflects nonlinear or interaction effects that the linear model cannot capture, but this was not tested directly.
- Feature importance describes what the model used, not what causes instability.

Figures are in `reports/figures/`: class counts, box plots by stability, correlation heatmap, feature importance, model comparison, and confusion matrices (initial and tuned versions). Result tables are saved as `reports/results_initial.csv` and `reports/results_tuned.csv`.

## Project Structure

```
GridGuard/
├── data/
│   ├── raw/            # original UCI file, never edited (not tracked by git)
│   └── processed/
├── notebooks/
│   └── gridguard_analysis.ipynb   # EDA, models, cross-validation, tuning
├── src/
│   ├── data_processing.py         # load data, features, leakage checks, split
│   ├── train.py                   # model definitions (initial and tuned settings)
│   ├── evaluate.py                # metrics and confusion counts
│   └── run.py                     # run the whole pipeline from the terminal
├── reports/
│   └── figures/
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

# run the pipeline (tuned settings; add --initial for the initial settings)
python -m src.run
```

The notebook (`notebooks/gridguard_analysis.ipynb`) expects `data/raw/Data_for_UCI_named.csv` and walks through the analysis step by step.

## Limitations

- The data is **simulated** and generated from fixed mathematical assumptions; features are sampled independently (near-zero correlation between them), unlike real grid telemetry.
- It covers **one four-node star grid**, so results will not automatically generalize to other grid structures or real grids.
- Test results come from **one train/test split and one random seed**. Cross-validation spreads are small, but no formal significance tests or confidence intervals have been computed.
- Models are not calibrated for real operational use, and this project does not include voltage, frequency, weather, demand, or time-series data.

## Status

Completed: data acquisition, inspection, EDA, leakage-safe preprocessing, three models with test-set evaluation, 5-fold cross-validation, hyperparameter tuning, feature importance, reusable `src/` pipeline.

Still to do: repeating the evaluation over several seeds, optional decision-threshold analysis, final report.

## Tools

Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, Jupyter, Git/GitHub.
