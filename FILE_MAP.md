# Карта файлов проекта Real Game of Life

Исследовательский проект по моделированию поведения клеток HeLa из данных time-lapse микроскопии.

---

## Корень проекта

| Файл | Описание |
|------|----------|
| `README.md` | Документация на RU/EN (36 KB) |
| `WORK_LOG.md` | Лог разработки (~50 ч, май 2026) |
| `FILE_MAP.md` | Этот файл |

---

## `GNN/` — Graph Neural Networks (398 MB)

Предсказание движения, формы, деления и гибели клеток.

### Модель

| Файл | Описание |
|------|----------|
| `gnn_model.py` | `CellInteractionGNN` — основная модель с головами: позиция, форма, деление, гибель |
| `gnn_layers.py` | Примитивы: `EdgeAwareAttentionConv`, `MLP`, `TemporalGRUCell` |
| `hybrid_field_gnn_model.py` | Экспериментальная модель с латентным пространственным полем |
| `spatial_field.py` | Геометрия поля для field-based моделей |

### Данные

| Файл | Описание |
|------|----------|
| `graph_conversion.py` | Конвертация таблиц → PyG-графы (radius, kNN, adaptive edges) |
| `graph_dataset.py` | `GraphDataset` — загрузка, сплиты, нормализация; таргеты позиции, формы и поляризации (`ELLIPSE_THETA`/`ELLIPSE_ASPECTRATIO`, угол — нематический, период π, см. `wrap_nematic_delta`) |
| `dataset_cache.py` | `GraphCache` — кэширование графов в `.pt`, CLI |

### Обучение

| Файл | Описание |
|------|----------|
| `train_one_step.py` | Основной цикл обучения: лосс, метрики, AMP, чекпоинты |
| `train_field_sequence.py` | Sequence-обучение с truncated BPTT для field-моделей |
| `rollout.py` | Конвертация предсказаний в граф для multi-step роллаута |
| `run_full_one_step.py` | Оркестратор полного пайплайна |
| `tabular_baseline.py` | Бейзлайн: XGBoost / LightGBM / RandomForest |
| `robustness_sweep.py` | Прогон train_one_step по нескольким сидам/split-mode, агрегация метрик (mean/std/median/IQR) |
| `kinetic_baseline.py` | Физический baseline без обучения (persistent random walk / OU по скорости), метрики по горизонтам роллаута в формате `evaluate_rollout.py` |

### Прочее

| Файл/Папка | Описание |
|------------|----------|
| `tests/` | 11 тест-файлов, покрывают весь пайплайн |
| `runs/` | Чекпоинты и метрики экспериментов |
| `cache/` | Кэшированные графы (`.pt`) |
| `scripts/run_one_step_server.sh` | Запуск обучения на GPU-сервере |
| `gnn_model_example.py` | Примеры использования модели |
| `gnn_layers_example.py` | Примеры использования слоёв |
| `RUN_ONE_STEP.md` | Документация по one-step workflow |

---

## `HeLa_Database/` — Обработка биоданных (4.9 GB)

От сырых XML TrackMate до очищенных parquet-таблиц.

| Файл | Описание |
|------|----------|
| `trackmate_pipeline.py` | Парсинг XML TrackMate, линковка треков, временные признаки |
| `trackmate_statistics.py` | QC и статистика треков |
| `shape_division_analysis.py` | Форм-признаки из масок (площадь, выпуклость, радиальные дескрипторы) |
| `cell_division_prediction_model.py` | XGBoost для предсказания деления (горизонты 3/5/10 кадров) |
| `gen_cells_app.py` | Интерактивная визуализация клеток |

### Данные

| Папка/Файл | Описание |
|------------|----------|
| `shape_division_analysis_dynamic/spot_shape_division_dataset.parquet` | Основной датасет |
| `DynamicNuclearNet/` | Маски сегментации от DNN-модели |
| `H2BmCherry_timelapse_60h/` | Сырые данные 60-часового таймлапса |
| `division_prediction_model/` | Артефакты модели предсказания деления |
| `tests/` | 1 тест-файл для модели деления |

---

## `NCA/` — Neural Cellular Automata (41 MB)

Авторегрессионный роллаут масок клеток через локальные правила обновления.

| Файл | Описание |
|------|----------|
| `model.py` | `NCA` — 2-слойная свёртка, стохастическое маскирование обновлений |
| `train.py` | Обучение: MSE/BCE/BCE_Dice, режимы лосса, CLI |
| `utils.py` | Роллаут, метрики (IoU, Dice, mass conservation), чекпоинты |
| `visualize.py` | GIF-анимации роллаутов, кривые потерь, heatmaps |
| `dataset.py` | `NCAfieldDataset` — `.npy` файлы формата `[T, H, W, F]` |
| `workbench.ipynb` | Jupyter-блокнот для экспериментов |
| `README.md` | Гайд по модели, форматам, CLI |
| `requirements.txt` | Зависимости модуля |

### Прочее

| Папка | Описание |
|-------|----------|
| `tests/` | 3 тест-файла (модель, датасет, обучение) |
| `runs/` | Артефакты экспериментов |
| `data_v2/train`, `val`, `test` | Сплиты датасета в формате `.npy` |
| `scripts/make_cell6972_nca_gnn_gif.py` | Генерация сравнительных GIF: NCA vs GNN |

---

## Пайплайн данных

```
Time-lapse микроскопия
        ↓
trackmate_pipeline.py  (XML → треки)
        ↓
shape_division_analysis.py  (маски → форм-признаки)
        ↓
spot_shape_division_dataset.parquet
      ├──→ dataset_cache.py → frame_graphs.pt
      │         ↓
      │    train_one_step.py / train_field_sequence.py
      │         ↓
      │    rollout.py → multi-step предсказания
      │
      └──→ NCA: dataset.py → train.py → роллаут масок
```

---

## Стек технологий

- **ML**: PyTorch, PyTorch Geometric, XGBoost, LightGBM, scikit-learn
- **Данные**: pandas, pyarrow, numpy
- **Визуализация**: matplotlib, pillow, Jupyter
- **Тестирование**: pytest
- **VCS**: git (remote: github.com:Ilya-Stetskiy/Real_game_of_life)
