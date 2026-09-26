from evt_demo.data_generator import generate_graph_time_series
from evt_demo.pipeline import prepare_sample
from evt_demo.evaluation import compare_evt_ar


def test_detection_and_features():
    series = generate_graph_time_series(seed=7)
    sample, detection = prepare_sample(series)
    assert detection.detected_at is not None
    assert abs(detection.detected_at - series.event_onset) < 30
    assert sample.features.shape == (18, 6)
    assert sample.edge_index.shape[0] == 2
    comparison = compare_evt_ar(series, detection)
    assert (
        comparison.precision.shape
        == comparison.recall.shape
        == comparison.f1.shape
        == (2,)
    )
