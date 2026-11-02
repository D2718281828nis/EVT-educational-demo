"""n-D эталон без оракула: фон кончается раньше истинного начала события, и видны тревоги до него.

В notebooks/n-d/03 и 05 фон задан равным истинному началу (baseline_size = 932, "labeled_event_onset_oracle_assisted"):
детектор ищет тревогу только после 932, поэтому ложная тревога до события там невозможна по построению. Здесь те же
данные, тот же GraphEVTPipeline и те же четыре спецификации, но фон кончается на 932 (оракул, для сравнения), 800,
600 или 400 отсчёте. Для каждой спецификации записываются все декластеризованные тревоги (find_events): сколько из них
до начала события, попала ли первая тревога в событие, и ожидаемое по утверждению 2.18.1 число ложных тревог
на участке до события, (длина участка) / ARL_0, где ARL_0 = sum_{j=1..r} (1 - gamma)^(-j).

Запуск из корня демо: python scripts/non_oracle_check.py  →  results/n-d/03_evt_non_oracle.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from graph_evt_agent import EVTConfig, GraphConfig, GraphEVTPipeline, InputConfig
from graph_evt_agent.evt import find_events

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/generated/kuramoto_propagation_series.csv"
META = ROOT / "data/generated/synthetic_datasets_metadata.json"
SEED = 42
BASELINES = [932, 800, 600, 400]
# те же четыре спецификации, что в results/n-d/05_specification_review.csv
SPECS = [("persistence_1_ar", 0.85, 0.97, 1, "ar"), ("persistence_2_ar", 0.85, 0.97, 2, "ar"),
         ("conservative_persistence_2", 0.90, 0.99, 2, "ar"), ("persistence_1_dfa", 0.85, 0.97, 1, "dfa")]


def arl0(gamma: float, r: int) -> float:
    return float(sum((1 - gamma) ** -j for j in range(1, r + 1)))


def main() -> None:
    df = pd.read_csv(DATA)
    names = [c for c in df.columns if c.startswith("channel_")]
    x = df[names].to_numpy()
    meta = json.loads(META.read_text(encoding="utf-8"))["kuramoto_propagation"]
    onset, end = int(meta["event_onset"]), int(meta["event_onset"]) + int(meta["event_duration"])
    source = names[int(meta["source_channel"])]
    rows = []
    for baseline in BASELINES:
        for role, q, gamma, r, method in SPECS:
            cfg = EVTConfig(baseline_size=baseline, tail_quantile=q, alarm_probability=gamma, top_k=3, persistence=r)
            pipe = GraphEVTPipeline(cfg, graph=GraphConfig(method=method, edge_threshold=0.35, random_state=SEED),
                                    input_config=InputConfig(missing="error", require_regular_time=True))
            res = pipe.run_detailed(x)
            alarms = find_events(res.detection, cfg)
            pre = [a for a in alarms if a < onset]
            first = res.detection.time_index
            rows.append({"baseline_end": baseline, "oracle": baseline == onset, "role": role, "tail_quantile": q,
                         "alarm_probability": gamma, "persistence": r, "graph_method": method,
                         "pre_onset_samples": onset - baseline, "arl0": round(arl0(gamma, r), 1),
                         "expected_pre_onset_alarms": round((onset - baseline) / arl0(gamma, r), 2),
                         "pre_onset_alarms": len(pre), "all_alarms": len(alarms),
                         "first_alarm": first, "first_alarm_in_event": first is not None and onset <= first < end,
                         "first_alarm_in_event_after_onset": next((a for a in alarms if onset <= a < end), None),
                         "estimated_source": None if res.ranking is None else names[res.ranking.source],
                         "source_correct": res.ranking is not None and names[res.ranking.source] == source})
    out = pd.DataFrame(rows)
    path = ROOT / "results/n-d/03_evt_non_oracle.csv"
    out.to_csv(path, index=False)
    print(out[["baseline_end", "role", "arl0", "expected_pre_onset_alarms", "pre_onset_alarms", "first_alarm",
               "first_alarm_in_event", "estimated_source", "source_correct"]].to_string(index=False))
    print(f"\n→ {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
