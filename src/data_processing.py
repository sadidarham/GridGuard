"""Data loading and preparation for GridGuard."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TARGET = "stabf"
LEAKAGE_COLUMN = "stab"      # directly determines stabf, so never an input
REDUNDANT_COLUMN = "p1"      # equals abs(p2 + p3 + p4)
FEATURES = ["tau1", "tau2", "tau3", "tau4", "p2", "p3", "p4", "g1", "g2", "g3", "g4"]

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "Data_for_UCI_named.csv"


def load_raw(path=DEFAULT_PATH):
    """Read the raw UCI CSV without modifying it."""
    return pd.read_csv(path)


def make_xy(df):
    """Return the 11 input features and the 0/1 target (1 = unstable)."""
    X = df[FEATURES]
    y = (df[TARGET] == "unstable").astype(int)
    assert LEAKAGE_COLUMN not in X.columns, "stab must never be an input"
    assert REDUNDANT_COLUMN not in X.columns
    return X, y


def split_data(X, y, test_size=0.2, random_state=RANDOM_STATE):
    """Stratified train/test split so both sets keep the class ratio."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)