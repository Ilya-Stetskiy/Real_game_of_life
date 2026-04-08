# NCA v1

Минимальный стенд для обучения и оценки Neural Cellular Automata на задаче динамики клеточной колонии.

## Что входит

- `dataset.py`: multi-file `.npy` dataset, split strategies, аугментации, collate для variable rollout horizon.
- `model.py`: baseline NCA с residual update, stochastic per-cell mask и опциональным alive mask.
- `train.py`: multi-step rollout training, deterministic baseline eval, stochastic eval.
- `utils.py`: сборка начального состояния, rollout helpers, checkpointing, метрики.
- `visualize.py`: heatmap, uncertainty map, metric curves и rollout animation.
- `workbench.ipynb`: удобный notebook для серверного запуска и контроля обучения.

## Форматы

- На диске: `numpy.ndarray` shape `[T, H, W, F_data]`
- В v1 `F_data = 1`, это только `density`
- В модели: `state = [B, F_model, H, W]`
- В v1 `F_model = 2`:
  - channel `0`: видимый `density`
  - channel `1`: latent hidden state

## Важный note

Первая версия intentionally minimalist:

- в данных только `1` видимый канал `density`
- latent hidden channel создаётся внутри модели и не хранится на диске
- hidden channel не супервизируется напрямую
- это baseline NCA
- следующий шаг: перейти к `F_model = 1 + hidden_dim` и при необходимости добавить learned hidden init

## Пути

Все CLI пути задаются относительными к корню репозитория. Базовые директории:

- `NCA/data`
- `NCA/runs`

## Локальная установка для CPU

```bash
python3 -m pip install -r NCA/requirements.txt
python3 -m pip install torch
```

GPU-сборку `torch` лучше ставить уже на сервере под конкретную CUDA.
