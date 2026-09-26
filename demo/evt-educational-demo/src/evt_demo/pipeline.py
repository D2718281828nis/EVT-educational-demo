"""End-to-end two-component graph extreme-event pipeline."""

from __future__ import annotations
from dataclasses import dataclass
from .data_generator import GraphTimeSeries, generate_graph_time_series
from .detection import DetectionResult, EVTEventDetector
from .features import edge_index, node_features
from .graph_model import CrossValidationResult, GraphSample, cross_validate_gat


@dataclass(frozen=True)
class PipelineResult:
    example: GraphTimeSeries
    detection: DetectionResult
    cv: CrossValidationResult


def prepare_sample(
    series: GraphTimeSeries, baseline_end: int = 700
) -> tuple[GraphSample, DetectionResult]:
    detection = EVTEventDetector(baseline_end).fit_detect(series.values)
    if detection.detected_at is None:
        raise RuntimeError("Event was not detected")
    features = node_features(
        series.values, series.adjacency, detection.detected_at, baseline_end
    )
    return (
        GraphSample(features, edge_index(series.adjacency), series.source_node),
        detection,
    )


def run_pipeline(
    n_scenarios: int = 18, seed: int = 42, epochs: int = 40
) -> PipelineResult:
    if n_scenarios < 6:
        raise ValueError("n_scenarios must be at least 6")
    samples = []
    example = None
    example_detection = None
    for index in range(n_scenarios):
        series = generate_graph_time_series(source_node=index % 18, seed=seed + index)
        sample, detection = prepare_sample(series)
        samples.append(sample)
        if index == 0:
            example, example_detection = series, detection
    cv = cross_validate_gat(samples, n_splits=3, epochs=epochs, seed=seed)
    assert example is not None and example_detection is not None
    return PipelineResult(example, example_detection, cv)
