# NCA: Neural Cellular Automata для динамики клеточных масок

Этот модуль содержит минимальный, но расширяемый baseline для обучения Neural Cellular Automata (NCA) на последовательностях клеточных масок или плотностей. Основная задача: по начальному наблюдаемому состоянию предсказывать дальнейшую пространственно-временную динамику через итеративный rollout одной локальной update-функции.

## Цель работы

Исследовать применимость Neural Cellular Automata для моделирования
пространственно-временной динамики клеточных масок.

Основной фокус:
- устойчивость rollout
- способность к обобщению
- влияние hidden state


## Идея подхода

NCA хранит состояние каждой клетки сетки как набор каналов:

```text
state_channels = data_channels + hidden_channels
```

- `data_channels` - наблюдаемые каналы, которые есть в `.npy` данных на диске.
- `hidden_channels` - latent channels, которые не лежат в данных и создаются моделью внутри начального состояния.
- `primary_channel` - главный наблюдаемый канал, по которому по умолчанию считаются публичные метрики и визуализации.
- `loss_channels` - какие наблюдаемые каналы supervised_loss: только `primary` или все `all_observed`.

Модель применяет один и тот же локальный сверточный update rule на каждом шаге rollout. Hidden channels не имеют прямого target и обучаются только косвенно через ошибку rollout по наблюдаемым каналам.

## Устройство модели NCA

NCA можно понимать как обучаемую клеточную автоматику. Модель не пытается сразу нарисовать весь будущий кадр. Вместо этого она много раз применяет одно и то же локальное правило обновления к состоянию сетки. Будущая динамика появляется из последовательности маленьких изменений.

Такой подход естественен для масок и плотностей, где глобальное поведение складывается из локальных взаимодействий: объект растёт, сжимается, сглаживается, разделяется, исчезает или меняет форму через изменения в соседних пикселях.

### Состояние как память системы

Каждая клетка сетки хранит не одно число, а небольшой вектор состояния. Часть этого состояния наблюдаемая и соответствует данным. Другая часть скрытая и нужна модели как внутренняя память.

Observed channels отвечают за то, что реально есть в данных: маску, плотность или дополнительные измеренные признаки. Hidden channels не размечаются вручную и не имеют прямого target. Они полезны только тогда, когда помогают лучше предсказывать будущие observed channels.

Например, по одной текущей маске не всегда понятно, должен ли объект расширяться, затухать или двигаться. Hidden state может выучить такие промежуточные признаки сам: локальную фазу процесса, направление изменения, накопленную активность или контекст окрестности.

### Локальность как ограничение

Главный inductive bias NCA - локальность. Каждая клетка обновляется на основе небольшой окрестности, а не всей картины сразу. Поэтому модель вынуждена строить сложное поведение через локальные правила, как в классических клеточных автоматах.

Это полезно, если изучаемый процесс действительно локален. Тогда модель меньше склонна запоминать глобальные шаблоны конкретных кадров и больше опирается на правило, которое можно применять в разных местах сетки.

### Rollout вместо one-step prediction

Один шаг NCA - это небольшое изменение текущего состояния. Длинный прогноз получается через rollout:

```text
initial state -> step 1 -> step 2 -> ... -> step T
```

На каждом шаге используется одна и та же модель с одними и теми же параметрами. Поэтому она учится не просто "нарисовать следующий кадр", а выучить правило динамики, которое можно применять многократно.

Это важное отличие от обычного one-step predictor. Если правило нестабильно, ошибки будут накапливаться при rollout. Поэтому качество нужно смотреть не только на одном шаге, но и на длинном горизонте.

### Наблюдаемое и скрытое

Target в данных существует только для observed channels. Hidden channels напрямую не сравниваются с target. Они похожи на внутреннюю рабочую память системы: визуально они могут быть неинтерпретируемы, но если помогают rollout оставаться устойчивым, обучение будет их использовать.

Слабая регуляризация hidden channels нужна только для того, чтобы скрытое состояние не разрасталось без необходимости.

### Стохастическое обновление

В NCA не обязательно обновлять все клетки одновременно. При стохастическом rollout часть клеток обновляется, а часть сохраняет прежнее состояние. Это похоже на асинхронное обновление в клеточных автоматах.

Такой режим работает как регуляризация. Модель не должна полагаться на идеально синхронный порядок обновлений и учится более устойчивому локальному правилу. Для deterministic evaluation стохастика выключается, чтобы получить воспроизводимый прогноз.

### Alive mask

Alive mask - это дополнительное ограничение, которое говорит модели, где динамика вообще разрешена. Концептуально это способ отделить активную область от пустого фона.

Для разреженных масок этот механизм может быть слишком жёстким: если объект должен появляться или расширяться из слабого сигнала, alive mask может помешать. Поэтому его лучше воспринимать как экспериментальную опцию, а не как обязательную часть подхода.

### Как понимать выход модели

Для continuous density можно использовать `MSE`: выход модели напрямую трактуется как значение плотности.

Для масок лучше думать о выходе как о вероятности принадлежности пикселя объекту. В режимах `bce` и `bce_dice` модель фактически выдаёт logits, которые затем переводятся в probability. Поэтому для анализа важно смотреть не только raw prediction, но и thresholded mask.

Именно thresholded view показывает, научилась ли модель восстанавливать структуру объектов, или она просто выдаёт размытое поле вероятностей.

## Формат данных

Данные на диске должны быть сохранены как `numpy.ndarray`:

```text
[T, H, W, F_data]
```

Где:

- `T` - число временных шагов.
- `H, W` - размер сетки.
- `F_data` - число наблюдаемых каналов.

Примеры:

- базовый режим: `[T, H, W, 1]`, `data_channels=1`, `hidden_channels=1`;
- multi-channel режим: `[T, H, W, 2]`, `data_channels=2`, `hidden_channels>=1`.

Важно: `data_channels` должен совпадать с последней размерностью файлов. Hidden channels в файлах не хранятся.

## Loss и метрики

Поддерживаются два независимых выбора.

### Какие каналы учить

```text
loss_channels = "primary" | "all_observed"
```

- `primary` - supervised loss только по `primary_channel`.
- `all_observed` - supervised loss по всем `data_channels`.

### Чем учить observed channels

```text
supervised_loss = "mse" | "bce" | "bce_dice"
```

- `mse` - backward-compatible режим, подходит для continuous density.
- `bce` - лучше подходит для бинарных/маскоподобных данных.
- `bce_dice` - более строгий режим для масок: штрафует размытие границ и foreground/background дисбаланс.

Для `bce` и `bce_dice` prediction интерпретируется как logits, а для визуализации и метрик переводится через `sigmoid`.

Дополнительные параметры:

- `bce_pos_weight` - вес положительного класса для BCE.
- `lambda_dice` - вес Dice-компоненты в `bce_dice`.
- `eval_threshold` - порог бинаризации для IoU/Dice/Precision/Recall.

Публичные deterministic/stochastic метрики по умолчанию считаются по `primary_channel`.

## Почему MSE может быть обманчивым

Для разреженных масок `MSE` часто выглядит слишком хорошо, даже если prediction визуально плохой. Модель может научиться выдавать сглаженную occupancy map: объекты примерно локализованы, но границы размыты, а фон слегка подсвечен.

Для таких данных лучше смотреть не только `rollout_mse`, но и:

- `dice`;
- `iou`;
- `precision`;
- `recall`;
- thresholded prediction в notebook.

## Сплиты и честность оценки

В проекте есть три базовых режима:

```text
split_mode = "within_file" | "by_file" | "by_group"
```

- `within_file` удобен для smoke-test, но часто слишком добрый: train/val берутся из одной траектории.
- `by_file` лучше, но может быть недостаточным, если файлы связаны по имени, например `pos17_q0`, `pos17_q1`, `pos17_q2`.
- `by_group` группирует файлы по ключу из имени. По умолчанию используется regex `pos(\d+)`, поэтому все `q*` одного `pos` попадают только в один split.

Для честной оценки желательно группировать split по биологически/пространственно независимой единице, например по `pos`, чтобы все `q*` одного `pos` попадали только в один split.


## Файлы проекта

- `model.py` - класс `NCA`: residual update, stochastic per-cell update mask, optional alive mask.
- `dataset.py` - чтение `.npy`, split strategies, normalizer, augmentations, collate для variable rollout horizon.
- `train.py` - rollout training, loss computation, deterministic/stochastic eval, CLI.
- `utils.py` - seed/device helpers, initial state, rollout, checkpointing, MSE/mass/binary metrics.
- `visualize.py` - сохранение triptych, metric curves, uncertainty heatmap, rollout animation.
- `workbench.ipynb` - notebook для локального/серверного эксперимента и визуальной диагностики.
- `tests/` - smoke tests для dataset/model/training/metrics.
- `data/` - локальная папка с `.npy` данными.
- `runs/` - локальная папка с checkpoint/history/visual artifacts.

## Работа через notebook

Откройте:

```text
NCA/workbench.ipynb
```

Notebook:

- сам находит `PROJECT_ROOT`;
- импортирует код только через `NCA.*`;
- сохраняет артефакты в `NCA/runs/workbench`;
- показывает inline:
  - loss curve;
  - raw probability triptych;
  - thresholded mask triptych;
  - uncertainty heatmap;
  - deterministic rollout gif;
  - Dice/IoU/Precision/Recall.

Ключевые параметры находятся в `CONFIG`:

```python
CONFIG = {
    "data_channels": 1,
    "hidden_channels": 1,
    "primary_channel": 0,
    "loss_channels": "primary",
    "supervised_loss": "bce_dice",
    "bce_pos_weight": 6.0,
    "lambda_dice": 0.5,
    "eval_threshold": 0.5,
}
```

Если данные continuous, начните с `supervised_loss="mse"`. Если данные похожи на маски, используйте `bce_dice`.

## CLI запуск

Команды запускайте из корня проекта:

```bash
cd ....../Game_of_life/Real_game_of_life
```

### Подготовка HeLa карт

Новый рекомендуемый формат для масок:

```text
[T, H, W, 1]
```

Нулевые hidden channels в `.npy` хранить не нужно: они добавляются моделью через `hidden_channels`.

```bash
python3 "HeLa_Database/HeLa клетки/Обработка данных для первого тестового стенда/build_nca_maps.py" \
  --source-root "HeLa_Database/HeLa клетки/DynamicNuclearNet/DynamicNuclearNet/tif_data" \
  --out-root NCA/data_v2 \
  --dataset dnn \
  --grid-size 8 \
  --channels density \
  --split-policy by_group \
  --split-ratios 0.6 0.2 0.2 \
  --overwrite
```

Если нужны дополнительные observed channels из TrackMate:

```bash
python3 "HeLa_Database/HeLa клетки/Обработка данных для первого тестового стенда/build_nca_maps.py" \
  --source-root "HeLa_Database/HeLa клетки/DynamicNuclearNet/DynamicNuclearNet/tif_data" \
  --out-root NCA/data_v2_trackmate \
  --dataset dnn \
  --grid-size 8 \
  --channels all \
  --split-policy by_group \
  --split-ratios 0.6 0.2 0.2 \
  --overwrite
```

Конвертер пишет `manifest.jsonl`, `qc.csv` и `summary.json` рядом с массивами.

### Старый backward-compatible режим `[T,H,W,1]`

```bash
python3 -m NCA.train \
  --data-root NCA/data_v2 \
  --out-dir NCA/runs/default \
  --data-channels 1 \
  --hidden-channels 1 \
  --primary-channel 0 \
  --split-mode by_group \
  --loss-channels primary \
  --supervised-loss mse
```

### Масочный режим для `[T,H,W,1]`

```bash
python3 -m NCA.train \
  --data-root NCA/data_v2 \
  --out-dir NCA/runs/bce_dice \
  --data-channels 1 \
  --hidden-channels 1 \
  --primary-channel 0 \
  --split-mode by_group \
  --loss-channels primary \
  --supervised-loss bce_dice \
  --bce-pos-weight 6.0 \
  --lambda-dice 0.5 \
  --eval-threshold 0.5
```

### Multi-channel режим `[T,H,W,2]`

```bash
python3 -m NCA.train \
  --data-root NCA/data_2ch \
  --out-dir NCA/runs/two_channel \
  --data-channels 2 \
  --hidden-channels 2 \
  --primary-channel 0 \
  --loss-channels all_observed \
  --supervised-loss bce_dice \
  --bce-pos-weight 6.0
```

## Рекомендованный workflow

1. Проверить формы данных:

```python
import numpy as np
from pathlib import Path

for path in sorted(Path("NCA/data_v2").glob("**/*.npy"))[:5]:
    arr = np.load(path, mmap_mode="r")
    print(path.name, arr.shape, arr.dtype, arr.min(), arr.max())
```

2. Запустить короткий smoke-test через notebook.

3. Посмотреть не только loss curve, но и thresholded triptych.

4. Если prediction размывает границы:

- перейти с `mse` на `bce_dice`;
- увеличить `bce_pos_weight`;
- проверить `Dice/IoU`;
- увеличить rollout horizon;
- проверить, нет ли слишком доброго split.

5. Для честной оценки использовать `split_mode="by_group"` по `pos`, а не split по соседним временным окнам.

## Локальная установка

Минимально:

```bash
python3 -m pip install -r NCA/requirements.txt
python3 -m pip install torch
```

Для GPU ставьте `torch` под конкретную CUDA/серверное окружение.

## Текущее состояние baseline

Backward-compatible defaults сохранены:

```text
data_channels = 1
hidden_channels = 1
primary_channel = 0
loss_channels = "primary"
supervised_loss = "mse"
```

Новый рекомендуемый режим для масок:

```text
supervised_loss = "bce_dice"
bce_pos_weight = 6.0
eval_threshold = 0.5
```

Этот режим не гарантирует идеальные границы, но делает smoothing-решение менее выгодным и даёт более честную диагностику через thresholded visualization и binary metrics.
