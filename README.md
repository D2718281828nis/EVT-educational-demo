# EVT Educational Demo

**RU:** Учебный репозиторий о том, как теория экстремальных значений (EVT) в
сочетании с графом зависимостей и GNN/GAT позволяет не только обнаружить
экстремальное событие, но и найти **его начальный источник** (канал в
многоканальном ряду или окно начала события в одномерном).

**EN:** An educational repository showing how Extreme Value Theory (EVT),
combined with a dependence graph and GNN/GAT, detects an extreme event *and*
locates its **initial source** (a channel in multichannel data, or the onset
window in a one-dimensional series).

> Все данные синтетические; результаты нельзя использовать для инженерных,
> финансовых или клинических решений. / All data are synthetic; results must
> not be used for engineering, financial, or clinical decisions.

## Содержимое / Contents

| Путь / Path | Описание / Description |
|---|---|
| [how-new-aaproach-EVT-works.md](how-new-aaproach-EVT-works.md) | Двуязычное описание нового подхода (1-D и n-D) / Bilingual description of the approach |
| [EVT_THEORY.md](EVT_THEORY.md) | Теоретические основы EVT / EVT theory background |
| [MISTRAL_VSCODE.md](MISTRAL_VSCODE.md) | Настройка LLM-режима в VS Code / LLM mode setup in VS Code |
| [demo/evt-educational-demo/](demo/evt-educational-demo/) | Код, данные и ноутбуки / Code, data, and notebooks |

## Подход в трёх шагах / The approach in three steps

1. **EVT-детекция / EVT detection** — робастная стандартизация по фону, POT/GPD-порог, момент устойчивого превышения. / robust baseline standardization, POT/GPD threshold, time of persistent exceedance.
2. **Граф / Graph** — гипотеза зависимости (AR, DFA, Kuramoto-вейвлет); в 1-D узлы — временные окна. / a dependence hypothesis (AR, DFA, Kuramoto-wavelet); in 1-D the nodes are time windows.
3. **GAT-локализация / GAT localization** — вероятности источника по узлам и оценка argmax. / source probabilities per node and an argmax estimate.

## Быстрый старт / Quick start

```bash
python -m venv demo/evt-educational-demo/.venv
source demo/evt-educational-demo/.venv/bin/activate
python -m pip install -r demo/evt-educational-demo/requirements.txt
python -m pip install --no-deps -e demo/evt-educational-demo

cd demo/evt-educational-demo
python scripts/generate_datasets.py --seed 42 --points 1500   # данные / data
jupyter lab                                                    # ноутбуки / notebooks
python -m pytest -q                                            # тесты / tests
```

Ноутбуки / Notebooks:

- `notebooks/1-d/` — одномерный ряд, 9 ноутбуков (включая временной граф GNN/GAT) / one-dimensional series, 9 notebooks (including temporal-graph GNN/GAT);
- `notebooks/n-d/` — многоканальный ряд, 10 ноутбуков (включая итоговую фигуру детекции и источника в `07`) / multichannel series, 10 notebooks (including the final detection-and-source figure in `07`).

Подробности, генерация данных и опциональный LLM-режим (`MISTRAL_API_KEY` только
через окружение) — в [README модуля](demo/evt-educational-demo/README.md). /
Details, data generation, and the optional LLM mode (`MISTRAL_API_KEY` via the
environment only) are in the [module README](demo/evt-educational-demo/README.md).

## Зависимости / Dependencies

EVT-конвейер берётся из [Graph-EVT-agent](https://github.com/D2718281828nis/Graph-EVT-agent) и не копируется в ноутбуки. / The EVT pipeline comes from Graph-EVT-agent and is not copied into the notebooks.
