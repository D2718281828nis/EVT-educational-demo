# Промпт: автономная учебная демонстрация EVT

## Роль и результат

Ты — senior Python-разработчик и исследователь статистики. Создай **автономный учебный модуль** по Extreme Value Theory (EVT) внутри существующего репозитория.

- Рабочая директория модуля: `demo/evt-educational-demo/`.
- Не изменяй расчётные конвейеры, данные и результаты в корневых `src/`, `data/`, `scripts/` и `reference_results/`.
- Python 3.10+; аудитория — магистранты, аспиранты и исследователи.
- Интерфейс, документация и пояснения в ноутбуках — на русском языке; имена Python API — на английском.
- Модуль должен работать офлайн на синтетических данных. Реальные данные — только опционально, без обязательной загрузки из сети.

Не ограничивайся генерацией каркаса: реализуй функции, тесты, исполняемые ноутбуки и воспроизводимые рисунки. Не коммить виртуальное окружение, кэш, большие бинарные данные или результаты выполнения ноутбуков.

## Требуемая структура

```text
demo/evt-educational-demo/
├── README.md
├── requirements.txt
├── pyproject.toml
├── data/
│   ├── synthetic_data.csv
│   └── README.md
├── src/evt_demo/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── evt_analysis.py
│   ├── ar_model.py
│   └── visualization.py
├── notebooks/
│   ├── 01_introduction_to_evt.ipynb
│   ├── 02_block_maxima_gev.ipynb
│   ├── 03_peaks_over_threshold_gpd.ipynb
│   └── 04_evt_vs_autoregression.ipynb
├── scripts/
│   └── run_demo.py
├── tests/
│   ├── test_data_generator.py
│   ├── test_evt_analysis.py
│   └── test_ar_model.py
└── figures/
    └── .gitkeep
```

Используй `src`-layout и абсолютные импорты вида `from evt_demo...`. Ноутбуки должны импортировать установленный editable-пакет, а не менять `sys.path`.

## 1. Генерация данных

В `data_generator.py` реализуй:

```python
def generate_time_series_with_extreme_events(
    n_points: int = 2_000,
    phi: float = 0.8,
    sigma: float = 1.0,
    n_extreme_events: int = 3,
    extreme_magnitude: float = 8.0,
    extreme_duration: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]: ...

def save_to_csv(
    time_series: np.ndarray,
    extreme_mask: np.ndarray,
    path: str | Path,
) -> None: ...
```

Требования:

1. База — стационарный AR(1): $x_t=\phi x_{t-1}+\varepsilon_t$, $\varepsilon_t\sim N(0,\sigma^2)$.
2. Проверяй: `n_points >= 100`, `abs(phi) < 1`, `sigma > 0`, положительную длительность и возможность непересекающегося размещения событий.
3. Используй `numpy.random.default_rng(seed)`, не глобальный `np.random.seed`.
4. Размещай события воспроизводимо, без пересечений и не в периоде разогрева. Амплитуда задаётся в единицах **безусловного** стандартного отклонения AR(1), $\sigma/\sqrt{1-\phi^2}$.
5. Маска должна отмечать только внедрённые точки. CSV: `timestamp,value,is_extreme`; `is_extreme` хранить как `0/1`.
6. Один и тот же seed и параметры должны давать побитово одинаковые массивы и CSV.

## 2. EVT-анализ

В `evt_analysis.py` создай dataclass-результаты (`GEVFit`, `GPDFit`) и класс `EVTAnalyzer`.

### 2.1. Block Maxima / GEV

```python
def fit_gev_block_maxima(
    self, data: ArrayLike, block_size: int = 100
) -> GEVFit: ...
```

- Проверяй одномерность и конечность данных, `block_size >= 2`, минимум 10 полных блоков.
- Отбрасывай неполный последний блок явно и записывай число отброшенных точек в результат.
- Оценивай GEV через `scipy.stats.genextreme.fit`.
- **Обязательно учти соглашение SciPy:** его shape `c` имеет знак, противоположный принятому в EVT параметру $\xi$; возвращай `xi = -c`, `mu = loc`, `sigma = scale`.
- Результат содержит параметры, максимумы и границы блоков; не возвращай модуль/«объект распределения» как состояние.

### 2.2. Peaks Over Threshold / GPD

```python
def fit_gpd_peaks_over_threshold(
    self,
    data: ArrayLike,
    threshold: float | None = None,
    threshold_percentile: float = 95.0,
) -> GPDFit: ...
```

- Используй `scipy.stats.genpareto`, а не `pareto`.
- Фитируй положительные превышения $y=x-u$ с фиксированным `loc=0`.
- Возвращай `threshold`, `exceedances`, их исходные индексы, `xi`, `sigma_u` и долю превышений.
- Требуй минимум 20 превышений; при меньшем числе выдавай понятную ошибку с советом снизить порог или увеличить выборку.

### 2.3. Возвратные уровни

```python
def calculate_gev_return_levels(
    self,
    fit: GEVFit,
    return_periods: Sequence[float] = (10, 50, 100, 500),
    *,
    confidence_level: float = 0.95,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> pd.DataFrame: ...
```

Используй

$$
x_T=\mu+\frac{\sigma}{\xi}\left(\{-\log(1-1/T)\}^{-\xi}-1\right),\quad \xi\ne0,
$$

и непрерывный предел Гумбеля при $|\xi|<10^{-6}$. Период $T$ измеряется в блоках. Доверительные интервалы оцени параметрическим bootstrap; при неуспешной оптимизации пропускай реплику, но требуй не менее 80% успешных реплик.

### 2.4. Детекция POT

```python
def detect_extremes_via_pot(
    self,
    data: ArrayLike,
    fit: GPDFit,
    tail_probability: float = 0.01,
) -> np.ndarray: ...
```

Для $x>u$ вычисляй условную вероятность хвоста fitted GPD и отмечай точку, если она не больше `tail_probability`. Точки ниже порога не отмечай. Не размечай целиком блоки по одному экстремальному максимуму.

### 2.5. Диагностика

Метод `diagnostics(fit)` возвращает чистые массивы для QQ-, PP-, density- и return-level-графиков. Численные расчёты не должны зависеть от Matplotlib.

## 3. Авторегрессионная базовая модель

В `ar_model.py` реализуй `AutoregressiveModel(order: int = 1)` поверх `statsmodels.tsa.ar_model.AutoReg`:

- `fit(data) -> Self`;
- `predict(data, steps=1) -> np.ndarray`;
- `calculate_residuals(data) -> np.ndarray` с выравниванием до длины исходного ряда и `NaN` в первых `order` позициях;
- `detect_anomalies_via_residuals(data, threshold_std=3.0) -> np.ndarray`;
- `forecast_with_confidence(data, steps=100, confidence_level=0.95)`.

Проверяй fitted-state, параметры и конечность входов. Порог остатков описывай как простую baseline-эвристику, а не как теоретически равноценную замену EVT. Не утверждай, что AR-модель обязательно требует нормальности для оценки коэффициентов; нормальность нужна для стандартной гауссовской интерпретации интервалов и $k\sigma$-порогов.

## 4. Визуализация

В `visualization.py` реализуй функции:

- `plot_time_series_with_extremes`;
- `plot_gev_distribution`;
- `plot_gpd_exceedances`;
- `compare_evt_vs_autoregression`;
- `plot_diagnostic_qq`;
- `plot_return_levels`;
- `plot_mean_excess`.

Каждая функция:

1. принимает `save_path: str | Path | None`;
2. возвращает `(fig, axes)` и не вызывает `plt.show()`;
3. создаёт родительский каталог при сохранении;
4. использует подписи и легенды на русском языке;
5. не мутирует входные массивы и не меняет глобальный стиль Matplotlib;
6. сохраняет PNG с `dpi=150`, `bbox_inches="tight"`.

В сравнительном графике считай precision/recall/F1 с `zero_division=0`. Отмечай **точки**, а не целые блоки, и явно называй сравнение учебным: результат зависит от порогов и не должен заранее гарантировать превосходство EVT.

## 5. Ноутбуки

Создай четыре небольших исполняемых ноутбука без дублирования библиотечного кода:

1. `01_introduction_to_evt.ipynb`: генерация, график, интуиция хвостов.
2. `02_block_maxima_gev.ipynb`: блоки, GEV, знак shape в SciPy, возвратные уровни.
3. `03_peaks_over_threshold_gpd.ipynb`: mean excess, выбор порога, GPD, условные tail probabilities.
4. `04_evt_vs_autoregression.ipynb`: честное сравнение POT-детекции и AR-residual baseline по precision/recall/F1.

В каждом ноутбуке должны быть:

- цели обучения и предпосылки;
- последовательные markdown-пояснения между короткими ячейками кода;
- фиксированный seed;
- интерпретация результата без заранее заданного вывода;
- ограничения метода;
- 3–5 вопросов для самопроверки;
- сохранение нужных рисунков в `figures/`.

Ноутбуки должны выполняться с чистым kernel командой `jupyter nbconvert --execute --inplace ...` после `pip install -e .`.

## 6. CLI и данные

`scripts/run_demo.py` одной командой должен:

1. сгенерировать `data/synthetic_data.csv`;
2. обучить GEV, GPD и AR;
3. посчитать return levels и метрики детекции;
4. сохранить не менее четырёх PNG в `figures/`;
5. вывести компактную таблицу результатов.

Все пути вычисляй относительно корня demo-модуля, а не текущего рабочего каталога. CLI должен работать из корня основного репозитория:

```bash
python demo/evt-educational-demo/scripts/run_demo.py --seed 42
```

CSV — единственный коммитящийся набор. Не добавляй `real_data.csv`, если нет легального, документированного источника и лицензии.

## 7. README и зависимости

README demo-модуля должен содержать:

- цели и границы применимости;
- установку из корня основного репозитория;
- быстрый запуск CLI и ноутбуков;
- дерево только demo-модуля;
- таблицу «GEV Block Maxima / GPD POT / AR residual baseline» без ложного тезиса, что один метод всегда лучше;
- пояснение параметров $\mu$, $\sigma$, $\xi$ и соглашения SciPy `c = -xi`;
- ссылки на Fisher–Tippett, Gnedenko, Coles (2001), de Haan & Ferreira (2006) и официальную документацию SciPy;
- раздел о воспроизводимости и ограничениях синтетического эксперимента.

В `requirements.txt` укажи совместимые нижние границы без копирования тяжёлых зависимостей основного проекта:

```text
numpy>=1.24
pandas>=2.0
scipy>=1.10
matplotlib>=3.7
scikit-learn>=1.3
statsmodels>=0.14
jupyter>=1.0
pytest>=7.0
```

Seaborn не добавляй, если он не используется. Метаданные пакета и pytest-конфигурацию размести в `pyproject.toml`.

## 8. Тесты и критерии приёмки

Покрой тестами:

- валидацию параметров генератора;
- воспроизводимость seed и CSV;
- отсутствие пересечений событий и корректность маски;
- форму результата и знак $\xi$ для GEV;
- `loc=0`, индексы и минимальное число превышений для GPD;
- устойчивую формулу возвратного уровня при $\xi\approx0$;
- формы, dtype и выравнивание результатов AR;
- сохранение каждого графика во временный каталог;
- smoke-test CLI.

Не используй хрупкий тест вида «каждая внедрённая точка обязана быть больше $5s$»: AR-шум может частично компенсировать импульс. Вместо этого проверяй сам внедрённый сигнал/маску, воспроизводимость и статистическое отличие совокупности экстремальных точек.

Перед завершением выполни из `demo/evt-educational-demo/`:

```bash
python -m pytest -q
python scripts/run_demo.py --seed 42
jupyter nbconvert --execute --to notebook --inplace notebooks/01_introduction_to_evt.ipynb
jupyter nbconvert --execute --to notebook --inplace notebooks/02_block_maxima_gev.ipynb
jupyter nbconvert --execute --to notebook --inplace notebooks/03_peaks_over_threshold_gpd.ipynb
jupyter nbconvert --execute --to notebook --inplace notebooks/04_evt_vs_autoregression.ipynb
```

Удаляй output-ячеек перед коммитом. Итог считается готовым, если тесты проходят, CLI создаёт CSV и рисунки, четыре ноутбука выполняются с чистого состояния, а изменения не затрагивают существующие три предметных примера репозитория.

## Формат итогового отчёта

В конце перечисли:

1. созданные файлы;
2. ключевые статистические решения и соглашения параметризации;
3. команды проверки и их результат;
4. известные ограничения;
5. подтверждение, что существующие brain/aviation/finance pipelines не изменены.
