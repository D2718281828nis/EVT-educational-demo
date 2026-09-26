"""Two-component EVT detection and graph-attention localization."""

from .data_generator import (
    GraphTimeSeries,
    generate_graph_time_series,
    generate_time_series_with_extreme_event,
    generate_time_series_with_extreme_events,
)
from .detection import DetectionResult, EVTEventDetector
from .graph_model import CrossValidationResult, GraphSample, cross_validate_gat
from .graph_model import TrainingTrace, train_gat_with_trace
from .legacy import AutoregressiveModel, EVTAnalyzer, GEVFit, GPDFit
from .pipeline import PipelineResult, prepare_sample, run_pipeline

__all__ = [
    "GraphTimeSeries",
    "generate_graph_time_series",
    "generate_time_series_with_extreme_event",
    "generate_time_series_with_extreme_events",
    "DetectionResult",
    "EVTEventDetector",
    "GraphSample",
    "CrossValidationResult",
    "cross_validate_gat",
    "TrainingTrace",
    "train_gat_with_trace",
    "PipelineResult",
    "prepare_sample",
    "run_pipeline",
    "EVTAnalyzer",
    "GEVFit",
    "GPDFit",
    "AutoregressiveModel",
]
