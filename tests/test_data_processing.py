from src.data_processing import FEATURES, load_raw, make_xy, split_data


def test_leakage_and_redundant_columns_excluded():
    assert "stab" not in FEATURES
    assert "stabf" not in FEATURES
    assert "p1" not in FEATURES


def test_p1_is_derived_from_consumer_power():
    df = load_raw()
    diff = (df["p1"] - (df["p2"] + df["p3"] + df["p4"]).abs()).abs().max()
    assert diff < 1e-9


def test_split_is_stratified_and_sized_correctly():
    X, y = make_xy(load_raw())
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_train) == 8000 and len(X_test) == 2000
    assert abs(y_train.mean() - y_test.mean()) < 0.001
    assert set(X_train.index).isdisjoint(X_test.index)