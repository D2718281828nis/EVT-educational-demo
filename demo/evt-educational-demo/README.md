# Two-component graph EVT demo

Автономное решение задачи: **детектировать момент экстремального события в многоканальном временном ряду и локализовать его источник на графе каналов**.

## Архитектура

```text
многоканальный ряд + граф
        │
        ├─ 1. робастные отклонения каналов → графовый индикатор → POT/GPD → момент события
        │
        └─ 2. признаки узлов около момента → GATv2 по рёбрам → вероятность узла-источника
                                                   └─ CV по независимым событиям
```

Компонент детекции агрегирует наиболее сильные робастные отклонения каналов и оценивает хвост фонового индикатора обобщённым распределением Парето. Компонент локализации получает амплитуду, энергию, задержку, степень узла и отклик соседей. GATv2 обучает коэффициенты внимания вдоль рёбер и ранжирует узлы как возможные источники. Кросс-валидация делит целые независимые реализации событий, поэтому каналы одного события не попадают одновременно в обучение и тест.

## Запуск

Из корня репозитория:

```bash
python3 -m venv demo/evt-educational-demo/.venv
source demo/evt-educational-demo/.venv/bin/activate
python -m pip install -e demo/evt-educational-demo
python demo/evt-educational-demo/scripts/run_demo.py --seed 42 --epochs 40
```

Быстрый smoke-run:

```bash
python demo/evt-educational-demo/scripts/run_demo.py --seed 42 --epochs 2 --scenarios 6
```

Результаты находятся в `demo/evt-educational-demo/results/`:

- `multichannel_example.csv` — пример ряда;
- `results.json` — задержка детекции, источник и метрики CV;
- `two_component_graph_method.png` — обе компоненты метода на одном рисунке.

## Ноутбуки

После установки dev-зависимостей (`python -m pip install -e 'demo/evt-educational-demo[dev]'`) последовательно откройте:

1. `01_multichannel_graph_data.ipynb` — синтез ряда и графа;
2. `02_evt_moment_detection.ipynb` — POT/GPD-детекция момента;
3. `03_node_features.ipynb` — признаки узлов;
4. `04_evt_vs_ar.ipynb` — отображение срабатываний и отдельный график precision/recall/F1;
5. `05_gnn_gat_training.ipynb` — граф, кривая обучения GNN и веса внимания GAT;
6. `06_evt_source_localization.ipynb` — визуальный поиск источника в синтезированном ряду.

Каждый актуальный ноутбук первой ячейкой автоматически добавляет локальный `src/` в путь импорта, поэтому `ModuleNotFoundError: No module named 'evt_demo'` не возникает даже без editable-установки. Если Jupyter уже держит старый модуль в памяти, выполните **Kernel → Restart Kernel → Run All**. Для совместимости со скачанными ранее ноутбуками пакет экспортирует прежние `EVTAnalyzer`, `AutoregressiveModel`, `generate_time_series_with_extreme_events` и функции базовой визуализации; актуальные ноутбуки используют графовый API `generate_graph_time_series`.

## Проверка

```bash
cd demo/evt-educational-demo
python -m pytest -q
```

## Ограничения

Данные синтетические, граф известен заранее, а события независимы. В прикладной задаче структуру графа, длину фона, POT-порог и протокол разбиения следует обосновывать предметно. Метрика top-1 дополняется mean reciprocal rank, поскольку несколько соседних узлов могут иметь похожий ранний отклик.
