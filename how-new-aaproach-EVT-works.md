# Как работает новый подход EVT / How the New EVT Approach Works

> **Главный результат / Main result:** найти исходный узел (канал), в котором
> зародилось экстремальное событие, и отделить его от каналов, куда событие
> распространилось позднее. / Find the initial node (channel) where an extreme
> event originated, and distinguish it from channels reached later by the event.

Новый подход объединяет классическую теорию экстремальных значений (EVT) с
моделью зависимости между временными рядами. EVT отвечает на вопрос **«когда
наблюдение стало экстремальным?»**, граф описывает допустимые пути
распространения, а GAT отвечает на вопрос **«где событие началось?»**. / The new
approach combines classical Extreme Value Theory (EVT) with a model of dependence
between time series. EVT answers **“when did the observation become extreme?”**,
the graph describes plausible propagation paths, and GAT answers **“where did
the event start?”**.

---

## 1. Одномерный временной ряд / One-dimensional time series

Пусть дан один ряд $x_t$, $t=1,\ldots,T$. Примеры: сигнал одного электрода,
температура одного двигателя или убыток одного актива. / Let $x_t$,
$t=1,\ldots,T$, be a single series, such as one electrode signal, one engine
temperature, or one asset loss.

### Шаг 0. Определить событие / Step 0. Define the event

1. Выбрать верхний хвост $x_t$, нижний хвост $-x_t$ или модуль отклонения
   $|x_t-m|$. / Choose the upper tail $x_t$, lower tail $-x_t$, or absolute
   deviation $|x_t-m|$.
2. Задать фоновый интервал, который заведомо предшествует исследуемому событию.
   Его нельзя выбирать с использованием будущих данных. / Define a baseline
   interval known to precede the event; it must not be selected using future
   observations.
3. Определить единицу события: одиночное превышение, кластер соседних
   превышений или максимум блока. / Define the event unit: one exceedance, a
   cluster of adjacent exceedances, or a block maximum.

### Шаг 1. **EVT-детекция.** / Step 1. **EVT detection.**

1. Проверить пропуски, выбросы измерительного тракта, тренд, сезонность и смену
   масштаба. / Check missing values, measurement artifacts, trend, seasonality,
   and scale changes.
2. Оценить фон устойчиво, например медианой $m$ и MAD $s$, и построить
   стандартизованный индикатор

   $$z_t=\frac{|x_t-m|}{\max(s,\varepsilon)}.$$

   Estimate the baseline robustly, for example using median $m$ and MAD $s$,
   and construct the standardized indicator above.
3. Для максимумов блоков подогнать GEV; для превышений высокого порога $u$
   подогнать GPD к $y_t=z_t-u>0$. Порог и параметры оцениваются только на
   фоновом/обучающем интервале. / Fit GEV to block maxima, or fit GPD to
   high-threshold excesses $y_t=z_t-u>0$. Estimate the threshold and parameters
   only from the baseline/training interval.
4. Учесть серийную зависимость: соседние превышения объединить в эпизоды или
   потребовать превышение порога в течение нескольких последовательных шагов.
   / Account for serial dependence by declustering adjacent exceedances or by
   requiring persistence over several consecutive time steps.
5. Первое устойчивое превышение EVT-порога определить как момент детекции
   $t_{\mathrm{EVT}}$. / Set the first persistent threshold exceedance as the
   detection time $t_{\mathrm{EVT}}$.

### Что можно заключить в 1-D? / What can be concluded in 1-D?

В одном ряду доступны детекция момента, величина экстремума, хвостовая
вероятность и возвратный уровень. Однако **локализовать источник среди разных
узлов невозможно**, потому что наблюдается только один канал. Единственный
наблюдаемый канал можно назвать местом регистрации, но нельзя отличить источник
от результата распространения из ненаблюдаемой системы. Поэтому стадии
**графового распространения** и **GAT-локализации** начинаются только при
$p\ge2$. / A single series supports detection time, extreme magnitude, tail
probability, and return-level estimation. It cannot localize a source among
nodes because only one channel is observed. The observed channel is a recording
location, not necessarily the physical origin. **Graph propagation** and **GAT
localization** therefore require $p\ge2$.

---

## 2. Зависимые $n$-мерные ряды / Dependent $n$-dimensional time series

Пусть одновременно наблюдаются $p$ каналов

$$\mathbf X_t=(X_{t,1},\ldots,X_{t,p}),$$

например контакты стерео-ЭЭГ. Один эпизод может сначала появиться в одном
контакте, а затем распространиться на соседние контакты. / Suppose $p$ channels
are observed simultaneously, for example stereo-EEG contacts. One episode may
first appear at one contact and then spread to neighboring contacts.

### Шаг 0. Проверка данных и разделение эпизодов / Step 0. Data checks and event splitting

1. Синхронизировать каналы, частоту дискретизации и пропуски. / Align channels,
   sampling rate, and missing observations.
2. Разделить данные на фон, обучение, валидацию и тест **по целым независимым
   эпизодам**, а не по точкам времени или каналам. / Split baseline, training,
   validation, and test data by **complete independent episodes**, not by time
   points or channels.
3. Все параметры стандартизации, EVT-порог и правила построения графа оценивать
   только по фону или обучающей части. / Estimate normalization, the EVT
   threshold, and graph-construction rules only from baseline or training data.

### 1. **EVT-детекция.** / **EVT detection.**

1. Для каждого канала по фону вычислить устойчивые $m_j$ и $s_j$, затем

   $$z_{t,j}=\frac{|X_{t,j}-m_j|}{\max(s_j,\varepsilon)}.$$

   For each channel, estimate robust baseline location $m_j$ and scale $s_j$,
   and calculate the standardized deviations above.
2. В каждый момент агрегировать несколько наиболее сильных отклонений, а не
   только абсолютный максимум:

   $$I_t=\operatorname{mean}\!\left(\operatorname{TopK}_j z_{t,j}\right).$$

   At every time step, aggregate several strongest channel deviations rather
   than only the single maximum.
3. Подогнать GPD к превышениям фонового индикатора $I_t$ над высоким порогом и
   получить порог тревоги $q_{\mathrm{EVT}}$. / Fit a GPD to baseline excesses
   of $I_t$ and obtain the alarm threshold $q_{\mathrm{EVT}}$.
4. С учётом требования устойчивости определить первое $t$ после фона, для
   которого $I_t>q_{\mathrm{EVT}}$, как $t_{\mathrm{EVT}}$. / With a persistence
   requirement, define the first post-baseline $t$ satisfying
   $I_t>q_{\mathrm{EVT}}$ as $t_{\mathrm{EVT}}$.

**Результат стадии / Stage output:** обнаружен общий момент экстремального
события, но источник ещё не определён. / A system-wide extreme-event time is
detected, but its source is not yet known.

### 2. **Графовое распространение.** / **Graph propagation.**

Цель — построить гипотезу графа $G=(V,E)$: узлы $V$ — каналы, а ребро
$(i,j)\in E$ означает, что между каналами допустима передача или совместное
распространение сигнала. **Граф является проверяемой гипотезой зависимости, а не
доказательством причинности.** / The goal is to construct a graph hypothesis
$G=(V,E)$: nodes are channels, and an edge means that signal transfer or joint
propagation is plausible. **The graph is a testable dependence hypothesis, not
proof of causality.**

#### 2.1. Проверить стохастическую стационарность / Check stochastic stationarity

Проверку выполнять отдельно по каналам и, при необходимости, в скользящих
окнах. Использовать совместно графики, ADF (нулевая гипотеза единичного корня),
KPSS (нулевая гипотеза стационарности), устойчивость среднего/дисперсии и
автокорреляцию. Одного теста недостаточно. / Check each channel, optionally in
rolling windows. Combine plots, ADF (unit-root null), KPSS (stationarity null),
stability of mean/variance, and autocorrelation. One test alone is insufficient.

Далее используется следующее **операционное правило выбора гипотезы графа**. / Use
the following **operational rule for selecting a graph hypothesis**:

- **Ряд приблизительно стационарен / Approximately stationary:** подогнать
  обычную авторегрессионную модель AR($p$) на фоновом интервале каждого канала.
  Порядок выбирать по AIC/BIC или временной валидации. Связь каналов оценивать по
  синхронности/лаговой корреляции инноваций AR; ребро оставлять только при
  устойчивости эффекта на обучающих окнах. Если нужна явная направленная
  межканальная динамика, AR следует заменить заранее заявленной VAR-моделью. /
  Fit an ordinary AR($p$) model to each channel's baseline, selecting order by
  AIC/BIC or temporal validation. Form a connectivity hypothesis from
  contemporaneous or lagged dependence between AR innovations, retaining edges
  only when stable across training windows. If explicit directed cross-channel
  dynamics are required, replace separate AR models with a prespecified VAR.
- **Ряд нестационарен или режим меняется / Non-stationary or regime-changing:**
  не строить рёбра непосредственно по сырой корреляции. Проверить две
  альтернативные гипотезы — **(a) DFA** и **(b) осцилляторы Курамото с
  вейвлет-анализом**. / Do not build edges directly from raw correlation. Test
  two alternative hypotheses: **(a) DFA** and **(b) Kuramoto oscillators
  enhanced with wavelet analysis**.

#### 2.2(a). DFA-гипотеза / DFA hypothesis

1. Для каждого канала удалить локальные тренды на наборе масштабов и оценить
   флуктуационную функцию $F_j(n)$ и показатель
   $F_j(n)\propto n^{\alpha_j}$. / For each channel, detrend locally across a
   range of scales and estimate fluctuation curve $F_j(n)$ and exponent
   $F_j(n)\propto n^{\alpha_j}$.
2. Сравнить не только $\alpha_j$, но и профили $F_j(n)$ в скользящих окнах до и
   после $t_{\mathrm{EVT}}$. / Compare both $\alpha_j$ and the $F_j(n)$ profiles
   in rolling windows before and after $t_{\mathrm{EVT}}$.
3. Соединить каналы, если их многомасштабные профили устойчиво похожи или если
   cross-DFA показывает воспроизводимую совместную флуктуацию. Порог сходства
   выбрать только на обучающих эпизодах. / Connect channels when their
   multiscale profiles are stably similar, or when cross-DFA shows reproducible
   joint fluctuations. Select the similarity threshold using training episodes
   only.
4. Проверить устойчивость рёбер к диапазону масштабов, длине окна и
   декластеризации. / Test edge stability against scale range, window length,
   and declustering choices.

DFA устойчивее простой корреляции к некоторым трендам, но сходные показатели
$\alpha$ сами по себе не доказывают взаимодействие каналов. / DFA is more robust
than raw correlation to some trends, but similar $\alpha$ values alone do not
prove interaction.

#### 2.2(b). Гипотеза Курамото + вейвлеты / Kuramoto + wavelet hypothesis

1. Выполнить непрерывное вейвлет-преобразование каждого канала и выбрать
   частотно-временные полосы, связанные с исследуемым событием. Выбор полосы
   фиксировать по обучающим данным. / Apply a continuous wavelet transform and
   select event-relevant time-frequency bands using training data only.
2. Из комплексных коэффициентов получить мгновенную фазу $\phi_j(t,f)$. /
   Extract instantaneous phase $\phi_j(t,f)$ from complex coefficients.
3. Оценить парную фазовую синхронизацию и локальный параметр порядка Курамото,

   $$R(t,f)e^{i\Psi(t,f)}=\frac1p\sum_{j=1}^{p}e^{i\phi_j(t,f)}.$$

   Estimate pairwise phase locking and the local Kuramoto order parameter above.
4. Построить ребро между $i$ и $j$, если их фазовая связь выше порога, устойчива
   во времени и превышает уровень суррогатных данных. Лаг знака фазы можно
   использовать только как гипотезу направления. / Add an edge when phase
   coupling exceeds a threshold, is stable over time, and is stronger than a
   surrogate-data baseline. Phase lag may be used only as a directional
   hypothesis.
5. Проверить, объясняет ли модель связанных осцилляторов наблюдаемый переход от
   локального отклика к синхронизации соседей. / Check whether a coupled
   oscillator model explains the observed transition from local response to
   neighboring synchronization.

Вейвлет-фаза помогает при нестационарной частоте, однако общая внешняя сила или
объёмная проводимость также могут создать синхронизацию без прямого ребра. /
Wavelet phase is useful under time-varying frequency, but common forcing or
volume conduction can also induce synchronization without a direct edge.

#### 2.3. Выбрать и зафиксировать граф / Select and freeze the graph

Сравнить AR-, DFA- и Kuramoto-wavelet-гипотезы только на обучающих эпизодах по:
устойчивости рёбер, воспроизводимости между окнами, качеству прогноза/локализации
и согласованности с известной физической или анатомической структурой. После
выбора зафиксировать граф до тестирования. Можно также использовать ансамбль
графов, но правило объединения должно быть задано заранее. / Compare AR, DFA,
and Kuramoto-wavelet hypotheses only on training episodes using edge stability,
window-to-window reproducibility, predictive/localization performance, and
agreement with known physical or anatomical structure. Freeze the selected
graph before testing. A graph ensemble is also possible if its combination rule
is prespecified.

**Результат стадии / Stage output:** граф возможного распространения и для
каждого узла признаки отклика около $t_{\mathrm{EVT}}$: амплитуда, энергия,
наклон, задержка, степень и согласованность с соседями. / A candidate
propagation graph and per-node response features around $t_{\mathrm{EVT}}$:
amplitude, energy, slope, latency, degree, and neighbor consistency.

### 3. **GAT-локализация.** / **GAT localization.**

1. Сформировать один графовый пример на один независимый эпизод:
   $({\bf F},E,y)$, где ${\bf F}\in\mathbb R^{p\times d}$ — причинные узловые
   признаки, $E$ — зафиксированные рёбра, $y$ — известный источник. / Build one
   graph example per independent episode: $({\bf F},E,y)$, where
   ${\bf F}\in\mathbb R^{p\times d}$ contains causal node features, $E$ contains
   frozen edges, and $y$ is the known source.
2. Нормировать признаки статистиками только обучающих эпизодов. / Normalize
   features using training-episode statistics only.
3. Передавать сообщения GAT только по рёбрам $E$ и получить логит $\ell_j$ для
   каждого кандидата в источник. / Pass GAT messages only along $E$ and obtain a
   logit $\ell_j$ for every candidate source.
4. Преобразовать логиты в ранжирование


```math
P(y=j \mid \mathbf{F}, E) = \text{softmax}(\ell)_j = \frac{e^{\ell_j}}{\sum_{k} e^{\ell_k}}
```


   и ранжировать все узлы-кандидаты по этой вероятности. / Rank all candidate
   nodes by this probability.
7. Обучать по cross-entropy, а валидацию выполнять по целым эпизодам (при
   необходимости также по пациентам, объектам или электродным стволам). /
   Optimize cross-entropy and validate by complete episodes, additionally
   grouping by patient, object, or electrode shaft when required.
8. Сообщать Top-1 accuracy, mean reciprocal rank, Top-$k$, калибровку
   вероятностей и устойчивость результата к выбору графа. / Report Top-1
   accuracy, mean reciprocal rank, Top-$k$, probability calibration, and
   sensitivity to graph choice.

**Итог / Final output:**

$$
\widehat{s}=\arg\max_j P(y=j\mid {\bf F},E),
$$

где $\widehat{s}$ — оценка **начального источника экстремального значения**, а
полное распределение вероятностей показывает неопределённость локализации. /
Here $\widehat{s}$ estimates the **initial source of the extreme value**, while
the full probability distribution represents localization uncertainty.

---

## 3. Сводный алгоритм / End-to-end algorithm

```text
INPUT: one series x[t], or dependent channels X[t, j]

1. Reserve a baseline and split complete episodes into train/validation/test.
2. Robustly standardize using baseline statistics only.
3. Fit POT/GPD (or Block Maxima/GEV) on the baseline tail.
4. Detect the first persistent system-level extreme time t_EVT.

IF there is only one channel:
    report detection time, tail probability/return level, and uncertainty;
    source localization is not identifiable.
ELSE:
    test stochastic stationarity channel by channel.

    IF approximately stationary:
        fit ordinary AR models;
        propose edges from stable dependence between AR innovations.
    ELSE:
        graph hypothesis A = DFA/cross-DFA multiscale coupling;
        graph hypothesis B = wavelet phase coupling + Kuramoto dynamics.

    compare graph hypotheses on training episodes and freeze the choice.
    extract causal node features around t_EVT.
    train and validate GAT by complete episodes.
    return ranked source probabilities and argmax source.
```

## 4. Ограничения интерпретации / Interpretation limits

- EVT-порог означает статистическую редкость относительно выбранного фона, а не
  автоматически патологию или причинность. / An EVT threshold means statistical
  rarity relative to the selected baseline, not automatic pathology or causality.
- DFA-, AR- и Kuramoto-wavelet-графы — конкурирующие гипотезы зависимости. Их
  нельзя выбирать по тестовым меткам. / DFA, AR, and Kuramoto-wavelet graphs are
  competing dependence hypotheses and must not be selected using test labels.
- Внимание GAT не является доказательством причинного механизма. / GAT attention
  is not proof of a causal mechanism.
- Истинный источник должен быть определён независимо: экспериментальной меткой,
  клинической разметкой, временем инъекции синтетического события или иным
  внешним критерием. / Ground-truth source labels must come from an independent
  experimental, clinical, synthetic-injection, or other external criterion.
