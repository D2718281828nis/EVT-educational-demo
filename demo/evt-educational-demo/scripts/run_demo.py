#!/usr/bin/env python3
"""Run moment detection, GAT source localization, and event-level cross-validation."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evt_demo.pipeline import run_pipeline  # noqa: E402
from evt_demo.visualization import plot_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--scenarios", type=int, default=18)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = run_pipeline(args.scenarios, args.seed, args.epochs)
    example = result.example
    frame = pd.DataFrame(
        example.values,
        columns=[f"channel_{i:02d}" for i in range(example.values.shape[1])],
    )
    frame.insert(0, "timestamp", np.arange(len(frame)))
    frame.to_csv(output / "multichannel_example.csv", index=False)
    summary = {
        "true_onset": example.event_onset,
        "detected_at": result.detection.detected_at,
        "detection_delay": (
            None
            if result.detection.detected_at is None
            else result.detection.detected_at - example.event_onset
        ),
        "true_source": example.source_node,
        "predicted_source": int(np.argmax(result.cv.probabilities[0])),
        "fold_top1_accuracy": result.cv.fold_top1_accuracy.tolist(),
        "fold_mean_reciprocal_rank": result.cv.fold_mean_reciprocal_rank.tolist(),
    }
    (output / "results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fig, _ = plot_pipeline(result, output / "two_component_graph_method.png")
    import matplotlib.pyplot as plt

    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nРезультаты: {output}")


if __name__ == "__main__":
    main()
