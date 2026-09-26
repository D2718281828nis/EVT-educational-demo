"""Integration test for the checked-in dataset generator."""

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


def test_dataset_script_writes_every_artifact(tmp_path):
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(root / "data/generate_datasets.py"),
            "--points",
            "240",
            "--channels",
            "4",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    expected = {
        "fractal_extreme_series.csv",
        "fractal_extreme_series.png",
        "kuramoto_synchronized_series.csv",
        "kuramoto_synchronized_series.png",
        "synthetic_datasets_metadata.json",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected
    fractal = pd.read_csv(tmp_path / "fractal_extreme_series.csv")
    assert list(fractal) == ["timestamp", "value", "is_extreme"]
    assert len(fractal) == 240
    metadata = json.loads(
        (tmp_path / "synthetic_datasets_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["seed"] == 42
    assert metadata["kuramoto"]["n_channels"] == 4
