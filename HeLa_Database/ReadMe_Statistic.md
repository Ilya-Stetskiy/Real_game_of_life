# ReadMe_Statistic

Этот документ описывает статистическую обработку TrackMate XML после добавления жизненных циклов клеток.

Основной скрипт:

```text
trackmate_statistics.py
```

Он не меняет исходные XML. Скрипт читает TrackMate-разметку, повторяет ту же обработку, что и `trackmate_pipeline.py`, собирает жизненные циклы клеток и сохраняет статистические таблицы в CSV/JSON.

## Быстрый запуск

Из корня проекта `Real_game_of_life` через Anaconda Python:

```bat
D:\Anaconda3\NewAnaconda\python.exe "HeLa_Database\HeLa клетки\trackmate_statistics.py" --source-root "HeLa_Database\HeLa клетки" --out-dir "HeLa_Database\HeLa клетки\statistics_output" --keep-going
```

По умолчанию форма клетки дискретизируется 64 радиусами. Число лучей можно изменить:

```bat
D:\Anaconda3\NewAnaconda\python.exe "HeLa_Database\HeLa клетки\trackmate_statistics.py" --source-root "HeLa_Database\HeLa клетки" --out-dir "HeLa_Database\HeLa клетки\statistics_output_shape128" --shape-samples 128 --keep-going
```

Из WSL/bash:

```bash
"/mnt/d/Anaconda3/NewAnaconda/python.exe" "HeLa_Database/HeLa клетки/trackmate_statistics.py" \
  --source-root "HeLa_Database/HeLa клетки" \
  --out-dir "HeLa_Database/HeLa клетки/statistics_output" \
  --keep-going
```

Для одного XML:

```bash
"/mnt/d/Anaconda3/NewAnaconda/python.exe" "HeLa_Database/HeLa клетки/trackmate_statistics.py" \
  --source-root "HeLa_Database/HeLa клетки/DynamicNuclearNet/DynamicNuclearNet/tif_data/train/out/HeLa-S3_nuc_pos0_q3_5c/HeLa-S3_nuc_pos0_q3_5c_trackmate.xml" \
  --out-dir "HeLa_Database/HeLa клетки/statistics_one_xml"
```

## Что считается клеткой

В этой обработке клетка - это линейный участок lineage между событиями.

Правила:

- Если у spot-а ровно один следующий spot, и у следующего spot-а ровно один родитель, это та же клетка.
- Если у spot-а нет продолжения, жизненный цикл клетки заканчивается событием `death`.
- Если у spot-а несколько дочерних spot-ов, жизненный цикл текущей клетки заканчивается событием `division`.
- После `division` создаются новые дочерние клетки.
- Если у клетки нет известного родителя в XML, ее `generation = 0`.
- Если клетка родилась после деления, ее `generation = parent_generation + 1`.

Идентификаторы:

- `cell_id` - локальный id клетки внутри одного XML.
- `cell_uid` - глобальный id в статистике, включает id последовательности и `cell_id`.
- `parents_id` - локальные id родительских клеток через `|`.
- `childs_id` - локальные id дочерних клеток через `|`.
- `parents_uid` и `childs_uid` - такие же связи, но в глобальных id.

## Этапы обработки

1. Поиск XML

По умолчанию скрипт рекурсивно ищет `*.xml` внутри `--source-root`.

2. Парсинг TrackMate

Из XML читаются:

- `Spot`: `spot_id`, `frame`, `t`, `x`, `y`, `AREA`, `PERIMETER`, `CIRCULARITY`, `SOLIDITY`, `MEAN_INTENSITY_CH1`.
- `Edge`: `source`, `target`, `track_id`, `track_index`, `edge_speed`, `edge_displacement`, `edge_time`.

`track_id` берется из родительского узла `<Track>`, потому что в TrackMate XML сами `<Edge>` обычно не хранят `TRACK_ID`.

3. Фильтрация

По умолчанию используются:

```text
min_area = 50
max_area = 1000
```

Spot-ы вне диапазона удаляются. После этого удаляются edge-и, которые ссылаются на удаленные spot-ы.

Параметры можно изменить:

```bash
--min-area 30 --max-area 1500
```

4. Графовые признаки

Для каждого spot-а вычисляются:

- `next_ids`, `prev_ids`;
- `n_next`, `n_prev`;
- `has_next`, `has_prev`;
- `is_split`, `is_merge`;
- `dx`, `dy`, `speed` для линейного продолжения.

5. Жизненные циклы клеток

Скрипт строит таблицу клеток. Каждая строка - одна клетка от рождения до деления или смерти.

Для клетки считаются:

- начало и конец: `start_spot_id`, `end_spot_id`, `start_frame`, `end_frame`;
- поколение: `generation`;
- родительские и дочерние клетки: `parents_id`, `childs_id`;
- длительность: `lifetime_frames`;
- число наблюдений: `n_spots`;
- причина окончания: `end_reason`.

6. Параметрический контур формы

TrackMate XML хранит ROI-точки контура внутри тега `<Spot>...</Spot>`. Эти точки читаются как локальный контур относительно центра spot-а.

Для каждого spot-а строится фиксированный вектор формы:

```text
shape = [r(theta_0), r(theta_1), ..., r(theta_M-1)]
```

где `M = shape_samples`, по умолчанию `64`.

Для каждого угла выпускается луч из центра клетки, и сохраняется расстояние до пересечения с контуром. Это дает одинаковую длину признакового вектора для всех клеток и кадров.

Сохраняются две версии:

- `shape_r_000 ... shape_r_063` - абсолютные радиусы в пикселях;
- `shape_r_norm_000 ... shape_r_norm_063` - нормированные радиусы.

Нормировка:

```text
r_norm(theta) = r(theta) / sqrt(area / pi)
```

Так абсолютный размер остается в `AREA`, `PERIMETER`, `shape_mean_radius`, а `shape_r_norm_*` лучше описывает именно форму.

По умолчанию углы выравниваются по `ELLIPSE_THETA`, чтобы поворот клетки меньше влиял на вектор формы. Если нужны углы в координатах изображения, можно отключить выравнивание:

```bash
--no-shape-align
```

Контроль качества формы:

- `contour_point_count` - сколько ROI-точек было в XML;
- `shape_valid_rays` - сколько лучей нашли пересечение с контуром;
- `shape_missing_fraction` - доля лучей без прямого пересечения;
- `shape_contour_area` - площадь исходного ROI-полигона;
- `shape_area_ratio` - `shape_contour_area / AREA`;
- `shape_reconstruction_area` - площадь, восстановленная из radial-вектора;
- `shape_reconstruction_area_ratio` - `shape_reconstruction_area / AREA`;
- `shape_radius_cv` - относительная неоднородность радиусов.

Если `shape_missing_fraction` высокая, контур в этом кадре стоит считать менее надежным.

7. Соседство

По каждому кадру строится `cKDTree`. Для каждого spot-а считаются:

- `n_neighbors_r20`, `n_neighbors_r40`, `n_neighbors_r80`, `n_neighbors_r160` - число соседей на каждом масштабе;
- `density_r20`, `density_r40`, `density_r80`, `density_r160` - локальная плотность;
- `Fx_r*`, `Fy_r*`, `F_norm_r*` - взвешенный вектор соседского давления;
- `min_dist_r*`, `mean_dist_r*`, `free_space_r*` - расстояния до соседей и свободное пространство;
- `occupancy_area_r*` - доля площади окружения, занятая соседними клетками;
- `sector_r*_*` - секторные occupancy/count признаки вокруг клетки;
- `ring_count_20_40`, `ring_density_20_40` и аналогичные кольца между соседними радиусами;
- `density_grad_x_r*`, `density_grad_y_r*` - направленная асимметрия плотности;
- `distance_to_colony_edge`, `is_boundary_cell` - расстояние до края выпуклой оболочки колонии и флаг граничной клетки.

Старые колонки `n_neighbors`, `density`, `Fx`, `Fy` сохранены как алиасы масштаба `r40`, чтобы старые downstream-скрипты не ломались.

Параметры:

```bash
--neighbor-radii 20,40,80,160 --neighbor-decays 15,30,60,120 --n-sectors 8
```

8. Агрегация статистики

Spot-level признаки агрегируются до cell-level:

- средняя/медианная/min/max площадь;
- средняя интенсивность;
- средняя скорость;
- среднее число соседей;
- средняя локальная плотность;
- путь клетки;
- итоговое смещение;
- straightness = net_displacement / path_length.
- средние shape-метрики;
- средний нормированный shape-вектор `shape_r_norm_mean_000 ...`.

## Выходные файлы

В `--out-dir` создаются:

```text
dataset_summary.json
sequence_summary.csv
cell_lifecycle_statistics.csv
generation_summary.csv
sequence_generation_summary.csv
event_summary.csv
lineage_edges.csv
failed_xml.csv
```

`failed_xml.csv` появляется только если были ошибки и указан `--keep-going`.

### dataset_summary.json

Общий итог по всему набору:

- сколько XML обработано;
- сколько raw и filtered spot/edge;
- сколько клеток;
- сколько делений и смертей;
- максимальное поколение;
- распределение клеток по поколениям;
- контрольные счетчики ошибок.

Важные контрольные поля:

- `move_rows_with_missing_target` должно быть `0`;
- `spots_without_cell` должно быть `0`.

### sequence_summary.csv

Одна строка на XML:

- `raw_spots`, `raw_edges`;
- `filtered_spots`, `filtered_edges`;
- `removed_spots`, `removed_edges`;
- `cells`;
- `generation_max`;
- `division_cells`;
- `death_cells`;
- `mean_lifetime_frames`;
- `median_lifetime_frames`.

Эта таблица нужна, чтобы быстро увидеть проблемные последовательности.

### cell_lifecycle_statistics.csv

Главная таблица. Одна строка - один жизненный цикл клетки.

Ключевые колонки:

- `sequence_uid`, `sequence_name`, `dataset`, `source_split`;
- `cell_id`, `cell_uid`;
- `generation`;
- `parents_id`, `childs_id`;
- `parents_uid`, `childs_uid`;
- `start_frame`, `end_frame`, `lifetime_frames`;
- `end_reason`;
- `is_division`, `is_death`;
- `area_mean`, `area_median`, `area_min`, `area_max`;
- `speed_mean`, `edge_speed_mean`, `edge_speed_max`;
- `path_length`, `net_displacement`, `straightness`;
- `neighbor_count_mean`, `local_density_mean`;
- `n_neighbors_r*_mean`, `density_r*_mean`, `Fx_r*_mean`, `Fy_r*_mean`, `F_norm_r*_mean`;
- `ring_count_*_mean`, `ring_density_*_mean`;
- `occupancy_area_r*_mean`, `free_space_r*_mean`;
- `distance_to_colony_edge_mean`, `is_boundary_cell_mean`;
- `shape_mean_radius_mean`, `shape_radius_cv_mean`;
- `shape_missing_fraction_mean`, `shape_area_ratio_mean`;
- `shape_reconstruction_area_ratio_mean`;
- `shape_r_norm_mean_000 ... shape_r_norm_mean_063`;
- `valid_for_ml`.

`valid_for_ml` сейчас означает, что у клетки `n_spots >= min_track_length`.

### generation_summary.csv

Сводка по поколениям для каждого dataset/split:

- число клеток;
- число делений;
- число смертей;
- средняя длительность;
- средняя площадь;
- средняя скорость;
- средний путь;
- среднее смещение.

Эта таблица нужна, чтобы сравнивать поведение поколений.

### sequence_generation_summary.csv

То же, что `generation_summary.csv`, но отдельно для каждой последовательности.

Нужна для поиска XML, где одно поколение ведет себя нетипично.

### event_summary.csv

События по кадрам:

- dataset;
- split;
- sequence;
- generation;
- end_frame;
- end_reason;
- events.

Эта таблица нужна, чтобы смотреть, в какие кадры происходят деления и смерти.

### lineage_edges.csv

Граф родитель-дочерняя клетка:

- `parent_cell_uid`;
- `child_cell_uid`;
- `parent_generation`;
- `child_generation`.

Эта таблица удобна для построения lineage tree.

## Рекомендуемая проверка результата

После обработки проверь:

```text
dataset_summary.json
```

Нормальные признаки:

- `spots_without_cell = 0`;
- `move_rows_with_missing_target = 0`;
- `max_generation` больше 0, если в данных есть деления;
- `division_cells` совпадает с ожидаемым числом split-событий.
- `mean_shape_missing_fraction` близко к 0;
- `mean_shape_area_ratio` разумно близко к 1.

Для DynamicNuclearNet после текущей логики жизненных циклов было найдено 51 событие деления и максимальное поколение 7.

## Анализ формы перед будущим делением

Для проверки, связана ли форма клетки с будущим делением, используется:

```text
shape_division_analysis.py
```

Пример запуска только на DynamicNuclearNet:

```bat
D:\Anaconda3\NewAnaconda\python.exe "HeLa_Database\HeLa клетки\shape_division_analysis.py" --source-root "HeLa_Database\HeLa клетки\DynamicNuclearNet" --out-dir "HeLa_Database\HeLa клетки\shape_division_analysis_dynamic" --shape-samples 64 --horizons 3,5,10 --keep-going
```

Что делает скрипт:

- строит lifecycle каждой клетки и shape-вектор каждого spot-а;
- исключает терминальный кадр деления/смерти через `--lead-frames 1`;
- размечает события `division_within_3_frames`, `division_within_5_frames`, `division_within_10_frames`;
- сравнивает shape-признаки делящихся и неделящихся клеток через разницу средних и Cohen's d;
- обучает простую logistic regression с GroupKFold по XML-последовательностям, чтобы соседние кадры одной записи не попадали одновременно в train и validation.

Основные выходы:

- `shape_division_analysis_summary.json` - краткая сводка и top effects;
- `shape_division_effects.csv` - univariate effect size по признакам;
- `angular_shape_difference.csv` - где именно radial-контур отличается по углам;
- `shape_division_model_scores.csv` - качество shape/shape+size моделей;
- `spot_shape_division_dataset.parquet` - spot-level датасет с future labels.

Интерпретация текущего запуска по DynamicNuclearNet: зависимость есть, но сигнал сильно несбалансирован по классам. Наиболее сильный одиночный shape-признак - `SOLIDITY`: перед наблюдаемым делением он ниже. Радиальный контур сам по себе дает слабый сигнал; заметнее работают агрегированные shape-скаляры и размер клетки.

## Ограничения

- Поколение считается только по известной части lineage внутри XML. Если родитель был до первого кадра, клетка начинается с `generation = 0`.
- После фильтрации по площади некоторые связи могут исчезнуть. Это осознанно: статистика считается только по валидным spot-ам.
- `cell_id` локален для XML. Для объединенных таблиц лучше использовать `cell_uid`.
- Merge-события поддержаны в логике, но в проверенном DynamicNuclearNet после фильтрации merge-ов не было.
- Параметрический контур хорошо работает для форм, близких к star-convex. Если контур сильно вогнутый или центр выбран плохо, radial-вектор будет приближением.
- При соприкасающихся клетках и шумных ROI ошибка формы будет выше; для этого сохраняются `shape_missing_fraction` и area-ratio метрики.
