"""Run the GridGuard pipeline: load, split, train, evaluate on the test set."""
import argparse
from pathlib import Path

from src.data_processing import load_raw, make_xy, split_data
from src.evaluate import evaluate
from src.train import build_models


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate GridGuard models.")
    parser.add_argument("--initial", action="store_true",
                        help="use the initial (untuned) settings instead of the tuned ones")
    args = parser.parse_args()

    df = load_raw()
    X, y = make_xy(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    models = build_models(tuned=not args.initial)
    for model in models.values():
        model.fit(X_train, y_train)

    results = evaluate(models, X_test, y_test)
    print(results.round(3))

    out = Path("reports") / ("results_initial.csv" if args.initial else "results_tuned.csv")
    results.round(4).to_csv(out)
    print("Saved", out)


if __name__ == "__main__":
    main()