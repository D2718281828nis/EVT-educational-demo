#!/usr/bin/env python3
"""Generate the fractal-noise and synchronized Kuramoto example datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evt_demo.data_generator import generate_fractal_extreme_series, generate_kuramoto_time_series
from evt_demo.visualization import plot_fractal_extreme_series, plot_kuramoto_time_series


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hurst", type=float, default=0.75)
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--source-channel", type=int, default=0)
    parser.add_argument("--points", type=int, default=1_500)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "generated")
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fractal_values, event_mask = generate_fractal_extreme_series(args.points, args.hurst, args.seed)
    pd.DataFrame({"timestamp": np.arange(args.points), "value": fractal_values, "is_extreme": event_mask.astype(int)}).to_csv(output / "fractal_extreme_series.csv", index=False)
    fractal_figure, _ = plot_fractal_extreme_series(fractal_values, event_mask, args.hurst, output / "fractal_extreme_series.png")

    kuramoto = generate_kuramoto_time_series(args.points, args.channels, args.source_channel, args.seed)
    frame = pd.DataFrame(kuramoto.values, columns=[f"channel_{i:02d}" for i in range(args.channels)])
    frame.insert(0, "timestamp", np.arange(args.points) * kuramoto.time_step)
    frame.to_csv(output / "kuramoto_synchronized_series.csv", index=False)
    kuramoto_figure, _ = plot_kuramoto_time_series(kuramoto, output / "kuramoto_synchronized_series.png")

    metadata = {
        "seed": args.seed,
        "fractal_noise": {"hurst_exponent": args.hurst, "event_onset": int(np.flatnonzero(event_mask)[0]), "event_duration": int(event_mask.sum()), "event_magnitude_in_sigma": 8.0},
        "kuramoto": {"n_channels": args.channels, "source_channel": args.source_channel, "coupling": 2.0, "time_step": kuramoto.time_step, "initial_coherence": float(kuramoto.coherence[0]), "final_coherence": float(kuramoto.coherence[-1])},
    }
    (output / "synthetic_datasets_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    import matplotlib.pyplot as plt
    plt.close(fractal_figure)
    plt.close(kuramoto_figure)
    print(f"Generated datasets and plots in {output}")


if __name__ == "__main__":
    main()
