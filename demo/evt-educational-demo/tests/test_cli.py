from pathlib import Path
import os
import subprocess
import sys
import json


def test_cli_smoke(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/run_demo.py"),
            "--epochs",
            "1",
            "--scenarios",
            "6",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
    )
    result = json.loads((tmp_path / "results.json").read_text())
    assert result["detected_at"] is not None
    assert (tmp_path / "two_component_graph_method.png").exists()
    assert (tmp_path / "multichannel_example.csv").exists()
