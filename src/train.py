"""Model definitions for GridGuard."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data_processing import RANDOM_STATE


def build_models(tuned=True):
    """Return the three models, unfitted.

    tuned=False uses the initial settings; tuned=True uses the settings chosen
    by grid search with 5-fold cross-validation on the training set only.
    """
    if tuned:
        lr_c, tree_kw, forest_kw = 0.1, dict(max_depth=12, min_samples_leaf=20), dict(
            n_estimators=400, max_depth=None, min_samples_leaf=2)
    else:
        lr_c, tree_kw, forest_kw = 1.0, dict(max_depth=5, min_samples_leaf=20), dict(
            n_estimators=200, min_samples_leaf=5)

    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=lr_c, random_state=RANDOM_STATE, max_iter=1000)),
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, **tree_kw),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **forest_kw),
    }