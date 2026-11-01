"""Reference output of Graph-EVT-agent graph methods for tests/port_check.js.

Usage: GRAPH_EVT_AGENT=/path/to/Graph-EVT-agent python tests/make_graph_reference.py
Writes tests/fixtures/graph_methods_reference.json: the input series and, for every configuration,
the dependence matrix and adjacency computed by graph_evt_agent.graph.build_graph.
The surrogate test is left out: numpy's generator and the page's mulberry32 differ by design.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("GRAPH_EVT_AGENT", str(Path(__file__).resolve().parents[2] / "Graph-EVT-agent")))
from graph_evt_agent import GraphConfig  # noqa: E402
from graph_evt_agent.graph import build_graph  # noqa: E402

EDGES = {(0, 1): 5, (0, 2): 15, (1, 2): 10, (1, 4): 20, (2, 3): 5, (3, 4): 10}
CONFIGS = [
    {"method": "ar", "edge_threshold": 0.2},
    {"method": "ar", "edge_rule": "topk", "top_k_edges": 2},
    {"method": "corr", "edge_threshold": 0.1},
    {"method": "partial", "edge_threshold": 0.1},
    {"method": "xcorr", "max_lag": 20, "edge_threshold": 0.1},
    {"method": "xcorr", "max_lag": 20, "edge_threshold": 0.1, "prune_indirect": True, "lag_tolerance": 2},
    {"method": "pwxcorr", "max_lag": 20, "edge_rule": "topk", "top_k_edges": 2},
    {"method": "granger", "max_lag": 20, "lag_step": 5, "edge_threshold": 0.02},
    {"method": "granger", "max_lag": 20, "lag_step": 5, "ridge": 0.0, "edge_rule": "topk", "top_k_edges": 2},
]


def delayed_network(n=700, coupling=0.3, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 5))
    noise = rng.normal(size=(n, 5))
    for t in range(n):
        x[t] = noise[t]
        for (i, j), lag in EDGES.items():
            if t >= lag:
                x[t, j] += coupling * x[t - lag, i]
                x[t, i] += coupling * x[t - lag, j]
    return x


data = delayed_network()
cases = []
for config in CONFIGS:
    result = build_graph(data, GraphConfig(**config))
    cases.append({"config": config, "method": result.method,
                  "dependence": result.dependence.tolist(), "adjacency": result.adjacency.astype(int).tolist()})
out = Path(__file__).resolve().parent / "fixtures" / "graph_methods_reference.json"
json.dump({"generator": "graph_evt_agent.graph.build_graph", "data": data.tolist(), "cases": cases}, open(out, "w"))
print(f"wrote {out} with {len(cases)} cases")
