import pandas as pd

from main import DATA_DIR


def test_supplied_inputs_resolve_outside_repository(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in ("train.csv", "test.csv"):
        frame = pd.read_csv(DATA_DIR / name, nrows=2)
        assert len(frame) == 2
        assert {"UserId", "ProductId", "Rating"} <= set(frame.columns)
