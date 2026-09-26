import numpy as np
from evt_demo.data_generator import generate_graph_time_series
from evt_demo.graph_model import cross_validate_gat, train_gat_with_trace
from evt_demo.pipeline import prepare_sample


def test_event_level_cross_validation():
    samples = [
        prepare_sample(generate_graph_time_series(source_node=i, seed=10 + i))[0]
        for i in range(6)
    ]
    result = cross_validate_gat(samples, n_splits=3, epochs=1, seed=2)
    assert result.probabilities.shape == (6, 18)
    assert np.allclose(result.probabilities.sum(1), 1)
    assert result.fold_top1_accuracy.shape == (3,)
    trace = train_gat_with_trace(samples, epochs=1, seed=2)
    assert trace.losses.shape == (1,)
    assert trace.attention_edge_index.shape[0] == 2
    assert np.isclose(trace.probabilities.sum(), 1)
