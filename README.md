# 🇷🇺 README (RU)

## Real Game of Life

### Кратко (5 секунд)

Проект строит вычислительную модель поведения живых клеток HeLa по данным time-lapse микроскопии. Главная идея: представить клетки как динамическую систему, где каждая клетка взаимодействует с соседями, меняет форму, движется, погибает или делится. В отличие от классической "Игры жизни", правила здесь не задаются вручную, а извлекаются из реальных треков, формы и локального окружения клеток.

### Обзор (50 секунд)

Этот репозиторий объединяет подготовку биологических данных, анализ формы клеток и обучение моделей для прогноза клеточной динамики.

- Проект работает с обработанными TrackMate-таблицами: координаты, связи между клетками во времени, признаки формы, плотность окружения и метки деления.
- Данные переводятся в графы по кадрам: узлы соответствуют клеткам, рёбра описывают локальные пространственные взаимодействия.
- GNN-модель обучается на one-step задаче: предсказать следующий шаг движения и формы, а также события смерти и деления.
- Для редких событий деления добавлены horizon-targets: модель может оценивать риск деления в ближайшие 3, 5 и 10 кадров.
- Для проверки гипотез используются baseline-модели и top-k метрики, потому что обычная accuracy почти бесполезна при очень редких делениях.

### Возможности

- Преобразование обработанных клеточных таблиц в графы PyTorch Geometric.
- Обратимые и тестируемые конвертеры между табличными данными и графовым представлением.
- Автоматическое построение one-step regression targets для движения и формы.
- Событийные targets для смерти, деления и деления в горизонте нескольких кадров.
- Temporal-признаки по ancestor-ссылкам TrackMate: прошлые значения признаков и изменения относительно прошлого состояния.
- GNN для локальных взаимодействий между клетками.
- Полный training-run: cache, split, обучение, checkpoint, метрики, итоговый отчёт.
- Tabular baselines для проверки, есть ли сигнал в признаках до запуска тяжёлого GNN.
- Top-k метрики для редких событий: сколько реальных делений попало в самые рискованные клетки.
- Тесты для сборщика графов, cache, обучения, baseline и метрик.

### Ключевые идеи и улучшения

- **Граф вместо независимых клеток.** Клетка рассматривается не изолированно, а в контексте соседей, плотности и локальной геометрии.
- **One-step сейчас, horizon позже.** Базовая задача остаётся простой и проверяемой, но архитектура уже поддерживает прогноз на несколько кадров.
- **Rare-event metrics.** Деления встречаются редко, поэтому важнее average precision и top-k hits, а не accuracy.
- **Temporal context без утечки будущего.** Temporal-признаки строятся только через ancestor-ссылки в прошлые кадры.
- **Baseline перед усложнением.** Если простая tabular-модель не видит сигнал, GNN-результат надо интерпретировать осторожно.
- **Server-ready pipeline.** Cache и training-run можно воспроизводимо запускать на локальной машине и на GPU-сервере.

### Высокоуровневый pipeline

```text
Time-lapse микроскопия / выход TrackMate
        |
        v
Обработанная таблица клеток с треками, формой и событиями
        |
        v
Графовый cache по кадрам
        |
        +--> Проверка tabular baseline
        |
        v
One-step / horizon GNN обучение
        |
        v
Метрики, checkpoints и итоговый отчёт
```

Основной поток работы:

1. Подготовить или получить обработанный parquet/CSV с клетками, треками и признаками формы.
2. Собрать graph cache: один граф на кадр внутри последовательности.
3. Проверить распределение targets и baseline-метрики.
4. Запустить GNN training-run.
5. Анализировать `AP`, `top-k hits`, `pos_rmse`, `shape_rmse` и event confusion counts.

### Установка

Рекомендуется запускать из корня репозитория.

```bash
git clone git@github.com:Ilya-Stetskiy/Real_game_of_life.git
cd Real_game_of_life
```

Минимальные зависимости для GNN-части:

```bash
pip install torch torch-geometric pandas pyarrow scikit-learn pytest
```

Локально в текущей рабочей среде использовался Python:

```bash
/mnt/d/Anaconda3/NewAnaconda/python.exe
```

Проверка тестов:

```bash
python -m pytest -q GNN/tests
```

На Windows/WSL можно явно указать локальный Python:

```bash
"/mnt/d/Anaconda3/NewAnaconda/python.exe" -m pytest -q "D:/Proga/Game_of_life/Real_game_of_life/GNN/tests"
```

### Использование

#### 1. Собрать стандартный graph cache

```bash
python -m Real_game_of_life.GNN.dataset_cache \
  --source Real_game_of_life/HeLa_Database/shape_division_analysis_dynamic/spot_shape_division_dataset.parquet \
  --out Real_game_of_life/GNN/cache/frame_graphs_dynamic.pt \
  --edge-radius 40 \
  --split-mode by_position \
  --seed 17
```

#### 2. Собрать temporal graph cache

```bash
python -m Real_game_of_life.GNN.dataset_cache \
  --source Real_game_of_life/HeLa_Database/shape_division_analysis_dynamic/spot_shape_division_dataset.parquet \
  --out Real_game_of_life/GNN/cache/frame_graphs_temporal.pt \
  --edge-radius 40 \
  --split-mode by_position \
  --seed 17 \
  --temporal-lags 1,2,3,5,10 \
  --temporal-features x,y,AREA,SOLIDITY,shape_mean_radius,shape_radius_cv,n_neighbors,density
```

#### 3. Запустить smoke training-run

```bash
python -m Real_game_of_life.GNN.run_full_one_step \
  --preset smoke \
  --device cpu \
  --out-dir Real_game_of_life/GNN/runs/full_smoke
```

#### 4. Запустить серверное обучение

```bash
PYTHON=python \
DEVICE=cuda \
PRESET=server \
CACHE=Real_game_of_life/GNN/cache/frame_graphs_temporal.pt \
OUT_DIR=Real_game_of_life/GNN/runs/horizon_temporal_server \
bash Real_game_of_life/GNN/scripts/run_one_step_server.sh
```

#### 5. Запустить tabular baseline

```bash
python -m Real_game_of_life.GNN.tabular_baseline \
  --cache Real_game_of_life/GNN/cache/frame_graphs_temporal.pt \
  --out-dir Real_game_of_life/GNN/runs/tabular_temporal_division_h10 \
  --target division_h10
```

### Структура проекта

```text
Real_game_of_life/
  HeLa_Database/
    trackmate_pipeline.py
    trackmate_statistics.py
    shape_division_analysis.py
    shape_division_analysis_dynamic/
  GNN/
    graph_conversion.py
    graph_dataset.py
    dataset_cache.py
    gnn_layers.py
    gnn_model.py
    train_one_step.py
    run_full_one_step.py
    tabular_baseline.py
    scripts/
    tests/
  NCA/
    dataset.py
    model.py
    train.py
    visualize.py
```

### Технические детали (500 секунд)

#### Модель данных

Проект использует обработанные наблюдения клеток HeLa. Каждая строка описывает одну найденную клетку в одном кадре. Важные поля:

- идентификаторы последовательности и кадра;
- идентификатор клетки или spot;
- координаты центроида;
- связи TrackMate со следующими клетками;
- признаки формы;
- локальная плотность и признаки соседства;
- метки деления для нескольких будущих горизонтов.

GNN-код ожидает обработанную parquet-таблицу, например:

```text
HeLa_Database/shape_division_analysis_dynamic/spot_shape_division_dataset.parquet
```

Крупные бинарные артефакты намеренно не должны версионироваться. Сгенерированные graph cache файлы лежат в `GNN/cache/`, а результаты обучения - в `GNN/runs/`.

#### Построение графов

Каждый граф соответствует одному кадру одной последовательности.

- Узел = одна клетка в текущем кадре.
- Признаки узла = текущие скалярные признаки, радиальные признаки формы и optional temporal-признаки ancestors.
- Ребро = локальная пространственная связь между соседними клетками.
- Признаки ребра = `dx`, `dy`, расстояние и единичное направление.
- Target-тензоры графа = one-step regression targets и event labels.

Рёбра можно строить по радиусу и, при необходимости, с nearest-neighbor логикой. Серверный workflow по умолчанию использует `edge_radius=40`: это практичный компромисс между локальным контекстом и разреженностью графа.

#### Targets

Модель предсказывает несколько групп targets:

- `target_delta_pos`: one-step смещение центроида.
- `target_delta_shape`: one-step изменение радиальных признаков формы.
- `target_death`: исчезает ли клетка без следующей связи до финального кадра.
- `target_division`: имеет ли клетка больше одной дочерней клетки в следующем кадре.
- `target_division_within_3`, `target_division_within_5`, `target_division_within_10`: риск деления внутри будущего горизонта.

Regression targets валидны только для клеток, у которых есть ровно одна известная следующая клетка. Узлы с делением не принуждаются к одному усреднённому daughter target; они используются как event-примеры.

#### Temporal-признаки

Temporal-признаки опциональны. Они идут назад по parent-ссылкам TrackMate и добавляют прошлое состояние клетки к текущему узлу.

Для лага `N` сборщик может добавить:

- `temporal_lagN_has_ancestor`;
- `temporal_lagN_frame_gap`;
- `temporal_lagN_<feature>`;
- `temporal_lagN_delta_<feature>`.

Набор temporal-признаков по умолчанию намеренно небольшой: позиция, площадь, solidity, статистики радиуса формы, число соседей и плотность. Это не раздувает сразу все признаки и делает первые temporal-эксперименты интерпретируемыми.

#### Архитектура модели

GNN построена вокруг локальных взаимодействий клеток:

- входные encoder'ы узлов и рёбер;
- несколько message-passing слоёв;
- отдельные prediction heads для позиции, формы, смерти, one-step деления и horizon-деления;
- optional temporal-state компоненты, зарезервированные для будущих rollout-моделей.

Текущая основная задача обучения всё ещё one-step supervised learning. Так проще проверить targets и метрики перед переходом к длинным autoregressive rollouts.

#### Workflow обучения

`run_full_one_step.py` управляет полным циклом:

1. собрать или загрузить graph cache;
2. записать cache summary;
3. обучить модель из cache;
4. выбрать лучший checkpoint по validation loss;
5. оценить модель на test split;
6. записать отчёты и metadata окружения.

Основные outputs:

- `best.pt`;
- `last.pt`;
- `history.csv`;
- `history.json`;
- `run_summary.json`;
- `full_run_summary.json`;
- `environment.json`;
- `final_report.md`.

#### Метрики

Dataset сильно несбалансирован для предсказания деления. Accuracy может быть близкой к 1.0 даже тогда, когда модель не находит ни одного деления. Поэтому для событий важны:

- average precision;
- precision / recall / F1;
- true positives, false positives и false negatives;
- top-k hits, precision и recall.

Для предсказания движения и формы:

- `pos_rmse` измеряет качество прогноза смещения центроида;
- `shape_rmse` измеряет качество прогноза изменения формы.

#### Текущий эмпирический статус

На текущем split one-step деление встречается крайне редко. Horizon targets информативнее, но тоже разрежены. Локальный temporal-cache baseline для `division_h10` показал небольшой рост AP у logistic regression, но top-k hits на test split остались нулевыми. Это значит, что temporal-признаки технически стоит проверить в GNN, но задача деления всё ещё сильно зависит от class imbalance и чувствительности split.

### Ограничения

- В test split очень мало положительных примеров деления.
- Метрики деления могут сильно зависеть от стратегии split.
- Текущая GNN - это supervised one-step модель, а не полноценный autoregressive simulator.
- Temporal-признаки зависят от parent-ссылок TrackMate; пропущенные или шумные связи снижают их ценность.
- Крупные raw microscopy файлы, XML и graph cache файлы не подходят для обычных Git-коммитов.
- Проект сейчас больше ориентирован на воспроизводимые эксперименты, чем на packaged library API.

### Дальнейшая работа

- Улучшить split strategy для редких биологических событий.
- Добавить event-focused sampling или loss weighting для division horizons.
- Сравнить temporal GNN training с temporal tabular baseline.
- Добавить calibration и threshold selection для event heads.
- Реализовать multi-step rollout evaluation.
- Проверить альтернативы построения графа: adaptive radius, kNN, density-aware edges.
- Добавить более богатые shape encoders вместо одних hand-crafted radial features.
- Сделать визуальную диагностику predicted division risk во времени.

---

# 🇬🇧 README (EN)

## Real Game of Life

### TL;DR (5 sec)

This project builds a computational model of HeLa cell behavior from time-lapse microscopy data. The core idea is to treat cells as a dynamic interacting system where each cell moves, changes shape, dies or divides. Unlike the classical Game of Life, the rules are not hand-written; they are learned from real tracks, shape features and local cell neighborhoods.

### Overview (50 sec)

The repository combines biological data processing, cell-shape analysis and machine learning models for cellular dynamics.

- The data comes from processed TrackMate-style tables: coordinates, temporal links, shape descriptors, local density and division labels.
- Each frame is converted into a graph: nodes are cells and edges represent local spatial interactions.
- The GNN is trained on a one-step task: predict the next movement and shape change, plus death and division events.
- Because division is rare, the project also defines horizon targets: division within the next 3, 5 and 10 frames.
- Baseline models and top-k metrics are included, because plain accuracy is misleading for rare biological events.

### Features / Capabilities

- Conversion from processed cell tables to PyTorch Geometric graphs.
- Tested data-to-graph and graph-to-training-data conversion logic.
- Automatic one-step regression targets for cell movement and shape change.
- Event targets for death, one-step division and horizon division risk.
- Temporal ancestor features from TrackMate links: previous values and current-minus-past deltas.
- A GNN model for local cell-cell interactions.
- Full training runs with cache creation, splits, checkpoints, metrics and final reports.
- Tabular baselines for checking whether a signal exists before running heavier GNN experiments.
- Top-k metrics for rare events: how many true divisions appear among the highest-risk cells.
- Tests for graph construction, cache generation, training, baselines and metrics.

### Key Ideas / Improvements

- **Graphs instead of independent cells.** A cell is modeled in the context of its neighbors, local density and spatial geometry.
- **One-step now, horizon-ready later.** The main task stays simple and verifiable while the model already supports future-horizon targets.
- **Rare-event metrics.** Division is too rare for accuracy to be meaningful, so average precision and top-k hits matter more.
- **Temporal context without future leakage.** Temporal features follow ancestor links backward in time only.
- **Baselines before complexity.** If simple tabular models cannot see a signal, GNN results should be interpreted carefully.
- **Server-ready workflow.** The cache and training pipeline can run both locally and on a GPU server.

### High-level Pipeline

```text
Time-lapse microscopy / TrackMate output
        |
        v
Processed cell table with tracks, shape and event labels
        |
        v
Frame-level graph cache
        |
        +--> Tabular baseline sanity checks
        |
        v
One-step / horizon GNN training
        |
        v
Metrics, checkpoints and final report
```

Typical workflow:

1. Prepare or obtain a processed parquet/CSV table with cell tracks and shape features.
2. Build a graph cache: one graph per frame inside each sequence.
3. Check target distributions and baseline metrics.
4. Run the GNN training pipeline.
5. Analyze `AP`, `top-k hits`, `pos_rmse`, `shape_rmse` and event confusion counts.

### Installation

Run commands from the repository root.

```bash
git clone git@github.com:Ilya-Stetskiy/Real_game_of_life.git
cd Real_game_of_life
```

Minimal dependencies for the GNN pipeline:

```bash
pip install torch torch-geometric pandas pyarrow scikit-learn pytest
```

The local development setup used this Python executable:

```bash
/mnt/d/Anaconda3/NewAnaconda/python.exe
```

Run tests:

```bash
python -m pytest -q GNN/tests
```

On Windows/WSL, the local Python can be called explicitly:

```bash
"/mnt/d/Anaconda3/NewAnaconda/python.exe" -m pytest -q "D:/Proga/Game_of_life/Real_game_of_life/GNN/tests"
```

### Usage

#### 1. Build the standard graph cache

```bash
python -m Real_game_of_life.GNN.dataset_cache \
  --source Real_game_of_life/HeLa_Database/shape_division_analysis_dynamic/spot_shape_division_dataset.parquet \
  --out Real_game_of_life/GNN/cache/frame_graphs_dynamic.pt \
  --edge-radius 40 \
  --split-mode by_position \
  --seed 17
```

#### 2. Build the temporal graph cache

```bash
python -m Real_game_of_life.GNN.dataset_cache \
  --source Real_game_of_life/HeLa_Database/shape_division_analysis_dynamic/spot_shape_division_dataset.parquet \
  --out Real_game_of_life/GNN/cache/frame_graphs_temporal.pt \
  --edge-radius 40 \
  --split-mode by_position \
  --seed 17 \
  --temporal-lags 1,2,3,5,10 \
  --temporal-features x,y,AREA,SOLIDITY,shape_mean_radius,shape_radius_cv,n_neighbors,density
```

#### 3. Run a smoke training job

```bash
python -m Real_game_of_life.GNN.run_full_one_step \
  --preset smoke \
  --device cpu \
  --out-dir Real_game_of_life/GNN/runs/full_smoke
```

#### 4. Run server training

```bash
PYTHON=python \
DEVICE=cuda \
PRESET=server \
CACHE=Real_game_of_life/GNN/cache/frame_graphs_temporal.pt \
OUT_DIR=Real_game_of_life/GNN/runs/horizon_temporal_server \
bash Real_game_of_life/GNN/scripts/run_one_step_server.sh
```

#### 5. Run a tabular baseline

```bash
python -m Real_game_of_life.GNN.tabular_baseline \
  --cache Real_game_of_life/GNN/cache/frame_graphs_temporal.pt \
  --out-dir Real_game_of_life/GNN/runs/tabular_temporal_division_h10 \
  --target division_h10
```

### Project Structure

```text
Real_game_of_life/
  HeLa_Database/
    trackmate_pipeline.py
    trackmate_statistics.py
    shape_division_analysis.py
    shape_division_analysis_dynamic/
  GNN/
    graph_conversion.py
    graph_dataset.py
    dataset_cache.py
    gnn_layers.py
    gnn_model.py
    train_one_step.py
    run_full_one_step.py
    tabular_baseline.py
    scripts/
    tests/
  NCA/
    dataset.py
    model.py
    train.py
    visualize.py
```

### Technical Details (500 sec)

#### Data model

The project uses processed HeLa cell observations. Each row describes one detected cell in one frame. Important fields include:

- sequence and frame identifiers;
- cell or spot id;
- centroid coordinates;
- TrackMate next links;
- shape descriptors;
- local density and neighbor features;
- division labels for several future horizons.

The GNN code expects a processed parquet table such as:

```text
HeLa_Database/shape_division_analysis_dynamic/spot_shape_division_dataset.parquet
```

Large binary artifacts are intentionally not meant to be versioned. Generated graph caches live under `GNN/cache/`, and training outputs live under `GNN/runs/`.

#### Graph construction

Each graph corresponds to one frame of one sequence.

- Node = one cell in the current frame.
- Node features = current scalar features, shape radii and optional temporal ancestor features.
- Edge = local spatial relation between nearby cells.
- Edge features = `dx`, `dy`, distance and unit direction.
- Graph target tensors = one-step regression targets and event labels.

Edges can be built with a radius threshold and optional nearest-neighbor logic. The default server workflow uses `edge_radius=40`, which is a practical compromise between local context and graph sparsity.

#### Targets

The model predicts several target groups:

- `target_delta_pos`: one-step centroid displacement.
- `target_delta_shape`: one-step shape change for radial shape features.
- `target_death`: whether a cell disappears without a next link before the final frame.
- `target_division`: whether a cell has more than one child in the next frame.
- `target_division_within_3`, `target_division_within_5`, `target_division_within_10`: division risk within a future horizon.

Regression targets are valid only when a cell has exactly one known next cell. Split nodes are not forced into a single daughter target; they are treated as event examples.

#### Temporal features

Temporal features are optional. They follow TrackMate parent links backwards and add past state to the current node.

For lag `N`, the builder can add:

- `temporal_lagN_has_ancestor`;
- `temporal_lagN_frame_gap`;
- `temporal_lagN_<feature>`;
- `temporal_lagN_delta_<feature>`.

The default temporal feature set is intentionally small: position, area, solidity, shape radius statistics, neighbor count and density. This avoids immediately doubling every feature and keeps the first temporal experiments interpretable.

#### Model architecture

The GNN is designed around local cell interactions:

- input node and edge encoders;
- stacked message-passing layers;
- separate prediction heads for position, shape, death, one-step division and division horizons;
- optional temporal-state components reserved for future rollout-style models.

The current main training task is still one-step supervised learning. This makes the target definition and metrics easier to verify before moving to long autoregressive rollouts.

#### Training workflow

`run_full_one_step.py` orchestrates the full cycle:

1. build or load a graph cache;
2. write cache summary;
3. train from cache;
4. select the best checkpoint by validation loss;
5. evaluate on the test split;
6. write reports and environment metadata.

Outputs include:

- `best.pt`;
- `last.pt`;
- `history.csv`;
- `history.json`;
- `run_summary.json`;
- `full_run_summary.json`;
- `environment.json`;
- `final_report.md`.

#### Metrics

The dataset is highly imbalanced for division prediction. Accuracy can be close to 1.0 even when the model finds no divisions. For this reason, the important event metrics are:

- average precision;
- precision / recall / F1;
- true positives, false positives and false negatives;
- top-k hits, precision and recall.

For movement and shape prediction:

- `pos_rmse` measures centroid displacement quality;
- `shape_rmse` measures shape delta quality.

#### Current empirical status

On the current split, one-step division is extremely rare. Horizon targets are more informative but still sparse. A local temporal-cache baseline for `division_h10` showed a small AP improvement for logistic regression, but top-k hits on the test split remained zero. This means temporal features are technically useful enough to test in the GNN, but the division task remains dominated by class imbalance and split sensitivity.

### Limitations

- The dataset has very few positive division examples in the test split.
- Division metrics can vary strongly with the split strategy.
- The current GNN is supervised one-step learning, not a fully autoregressive simulator yet.
- Temporal features depend on TrackMate parent links; missing or noisy links reduce their value.
- Large raw microscopy files, XML and graph caches are not suitable for normal Git commits.
- The project currently focuses on reproducible experimentation rather than a packaged library API.

### Future Work

- Improve split strategy for rare biological events.
- Add event-focused sampling or loss weighting for division horizons.
- Compare temporal GNN training against the temporal tabular baseline.
- Add calibration and threshold selection for event heads.
- Implement multi-step rollout evaluation.
- Explore graph construction alternatives: adaptive radius, kNN, density-aware edges.
- Add richer shape encoders instead of using only hand-crafted radial features.
- Produce visual diagnostics for predicted division risk over time.
