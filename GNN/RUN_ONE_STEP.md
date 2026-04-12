# One-step GNN training run

This folder contains a full-cycle one-step training entry point:

```bash
python -m Real_game_of_life.GNN.run_full_one_step --preset server --device cuda
```

The runner builds the graph cache if it is missing, trains the model, evaluates the
best validation checkpoint on the test split, and writes a final report.

## Server run

From the repository root:

```bash
PYTHON=python DEVICE=cuda bash Real_game_of_life/GNN/scripts/run_one_step_server.sh
```

Useful overrides:

```bash
PYTHON=python \
DEVICE=cuda \
PRESET=large \
OUT_DIR=/path/to/runs/hela_one_step_large \
bash Real_game_of_life/GNN/scripts/run_one_step_server.sh \
  --epochs 800 \
  --batch-size 96 \
  --num-workers 8
```

## Local smoke run

```bash
/mnt/d/Anaconda3/NewAnaconda/python.exe -m Real_game_of_life.GNN.run_full_one_step \
  --preset smoke \
  --device cpu \
  --out-dir D:/Proga/Game_of_life/Real_game_of_life/GNN/runs/full_smoke
```

## Main outputs

- `best.pt`: checkpoint selected by validation loss.
- `last.pt`: checkpoint from the last completed epoch.
- `epoch_XXXX.pt`: periodic checkpoints when `checkpoint_every > 0`.
- `history.csv` and `history.json`: train, validation and test metrics.
- `run_summary.json`: model, split and metric summary.
- `full_run_summary.json`: run summary plus environment and cache metadata.
- `environment.json`: Python, PyTorch, CUDA and PyG versions.
- `final_report.md`: compact human-readable result.

## Metrics note

One-step division is a very rare event in the current split. Do not judge that
head by accuracy alone. The useful metrics are precision, recall, F1 and average
precision. Position and shape heads should be tracked by `pos_rmse` and
`shape_rmse`.
