import matplotlib
import numpy as np

matplotlib.use("Agg")

from graph_evt_agent import EVTConfig, GraphConfig, GraphEVTPipeline, InputConfig

from evt_demo.data_generator import generate_kuramoto_time_series
from evt_demo.visualization import plot_evt_source_result


def test_evt_source_plot_exposes_detection_and_ranking(tmp_path):
    series = generate_kuramoto_time_series(n_points=420, n_channels=6, seed=42)
    baseline_size = 100
    result = GraphEVTPipeline(
        EVTConfig(
            baseline_size=baseline_size,
            tail_quantile=0.9,
            alarm_probability=0.99,
            persistence=2,
        ),
        graph=GraphConfig(method="ar", random_state=42),
        input_config=InputConfig(missing="error"),
    ).run_detailed(series.values)

    assert result.detection.detected
    assert result.ranking is not None
    destination = tmp_path / "result.png"
    figure, axes = plot_evt_source_result(
        series.values,
        result,
        baseline_size=baseline_size,
        channel_names=[f"sensor {index}" for index in range(6)],
        output_path=destination,
    )

    assert destination.exists()
    assert "EVT detection" in axes[0].get_title()
    assert "Initial-source ranking" in axes[2].get_title()
    assert len(axes[2].patches) == 6
    figure.clear()


def test_evt_source_plot_rejects_mismatched_time_axis():
    series = generate_kuramoto_time_series(n_points=120, n_channels=3, seed=1)
    result = GraphEVTPipeline(EVTConfig(baseline_size=40), graph=GraphConfig(method="ar")).run_detailed(series.values)

    with np.testing.assert_raises_regex(ValueError, "timestamps"):
        plot_evt_source_result(series.values, result, timestamps=np.arange(3))
