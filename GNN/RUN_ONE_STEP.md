# One-step GNN training run

This folder contains a full-cycle one-step training entry point:

```bash
python -m Real_game_of_life.GNN.run_full_one_step --preset server --device cuda
```

Run commands that import `Real_game_of_life.*` from the directory that contains
the `Real_game_of_life` folder, for example `/mnt/d/Proga/Game_of_life` in the
current local layout.

The runner builds the graph cache if it is missing, trains the model, evaluates the
best validation checkpoint on the test split, and writes a final report.
The model predicts one-step position/shape dynamics plus one-step track disappearance/division
events and division horizon events such as `division_h3`, `division_h5`, and
`division_h10`. Use `prediction_to_next_graph` from `GNN.rollout` to rebuild a
one-step prediction as a model-ready graph for further rollout steps. For the
field-conditioned model, `field_prediction_to_next_graph` rebuilds the graph and
carries the predicted `field_next` into the next step.

Field models require graph caches built with physical `data.pos_xy` coordinates.
If an older cache fails with a `data.pos_xy` error, rebuild it:

```bash
python -m Real_game_of_life.GNN.dataset_cache \
  --out Real_game_of_life/GNN/cache/frame_graphs_dynamic_v2.pt
```

## Server run

From the directory that contains the `Real_game_of_life` folder:

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

## Temporal cache

The cache builder can add ancestor-history features before training. These
features follow TrackMate parent links only into earlier frames, so they do not
leak future target labels into the node features.

```bash
python -m Real_game_of_life.GNN.dataset_cache \
  --source Real_game_of_life/HeLa_Database/shape_division_analysis_dynamic/spot_shape_division_dataset.parquet \
  --out Real_game_of_life/GNN/cache/frame_graphs_temporal.pt \
  --edge-radius 40 \
  --split-mode by_position_event_balanced \
  --seed 17 \
  --temporal-lags 1,2,3,5,10 \
  --temporal-features x,y,AREA,SOLIDITY,shape_mean_radius,shape_radius_cv,n_neighbors,density
```

Train on that cache by overriding `CACHE`:

```bash
PYTHON=python \
DEVICE=cuda \
PRESET=server \
CACHE=Real_game_of_life/GNN/cache/frame_graphs_temporal.pt \
OUT_DIR=Real_game_of_life/GNN/runs/horizon_temporal_server \
bash Real_game_of_life/GNN/scripts/run_one_step_server.sh \
  --temporal-lags 1,2,3,5,10 \
  --temporal-features x,y,AREA,SOLIDITY,shape_mean_radius,shape_radius_cv,n_neighbors,density
```

## Local smoke run

```bash
cd /mnt/d/Proga/Game_of_life
/mnt/d/Anaconda3/NewAnaconda/python.exe -m Real_game_of_life.GNN.run_full_one_step \
  --preset smoke \
  --device cpu \
  --out-dir D:/Proga/Game_of_life/Real_game_of_life/GNN/runs/full_smoke
```

## Experimental field-conditioned GNN

`train_one_step.py` can train the experimental `field_gnn` model type. This
keeps the current one-step supervised setup, but conditions each cell prediction
on both GNN node embeddings and a local patch from a latent spatial field. In
this mode the field is initialized from zeros for each batch.

```bash
cd /mnt/d/Proga/Game_of_life
/mnt/d/Anaconda3/NewAnaconda/python.exe -m Real_game_of_life.GNN.train_one_step \
  --cache Real_game_of_life/GNN/cache/frame_graphs_dynamic.pt \
  --out-dir Real_game_of_life/GNN/runs/field_gnn_smoke \
  --model-type field_gnn \
  --field-channels 16 \
  --field-height 128 \
  --field-width 128 \
  --field-cell-size 4.0 \
  --device cpu
```

This one-step mode does not train recurrent field memory: `field_writer` and
`field_update` are frozen there. Use sequence training for full GNN+field memory.

For true sequence memory, use `train_field_sequence.py`. It groups cached frame
graphs by `sequence_uid`, sorts frames by `frame`, initializes one field per
sequence, and carries that field across frames. The current implementation uses
truncated BPTT with `--bptt-window 3` by default: gradients flow through three
frames, then the field state is detached to keep memory bounded.

```bash
cd /mnt/d/Proga/Game_of_life
/mnt/d/Anaconda3/NewAnaconda/python.exe -m Real_game_of_life.GNN.train_field_sequence \
  --cache Real_game_of_life/GNN/cache/frame_graphs_dynamic_v2.pt \
  --out-dir Real_game_of_life/GNN/runs/field_sequence_smoke \
  --field-channels 16 \
  --field-height 128 \
  --field-width 128 \
  --field-cell-size 1.0 \
  --bptt-window 3 \
  --device cpu
```

By default the sequence trainer derives field origin and cell size from the train
split and reports coverage for each split. To force explicit geometry, add
`--no-field-auto-geometry`. Use `--strict-field-coverage` to fail when coverage
falls below `--min-field-coverage` instead of only warning. For current
approximately 0..510 pixel coordinates, an explicit geometry such as
`--field-height 128 --field-width 128 --field-cell-size 4.0` is a reasonable
starting point.

## Differentiable autoregressive rollout training

Use `train_rollout_bptt.py` to train the plain GNN through its own predicted
graphs instead of feeding a real graph at every step. Gradients flow through
predicted positions, `shape_r_norm_*` features, and edge attributes across the
rollout window. Radius/kNN edges are rebuilt from predicted positions at every
step; the discrete neighbor selection itself is not differentiable.

Start from a compatible one-step checkpoint rather than training rollout
dynamics from scratch:

```bash
cd /mnt/d/Proga/Game_of_life
/mnt/d/Anaconda3/NewAnaconda/python.exe -m Real_game_of_life.GNN.train_rollout_bptt \
  --cache Real_game_of_life/GNN/cache/frame_graphs_dynamic_v2_5000_balanced.pt \
  --init-from Real_game_of_life/GNN/runs/gnn_baseline_local_5000_balanced/best.pt \
  --out-dir Real_game_of_life/GNN/runs/gnn_rollout_bptt3_local_5000_balanced \
  --rollout-steps 3 \
  --hidden-dim 32 \
  --layers 2 \
  --dropout 0.1 \
  --epochs 15 \
  --device cpu
```

The model architecture arguments must match `--init-from`. The current trainer
uses a fixed rollout node set and trains position and shape state losses only.
Cells without a single-track GT descendant are excluded from later-step losses;
division births, disappearance, and event-head losses are not yet part of this
rollout training path.

Compare baseline and rollout-trained checkpoints on the same held-out test
sequences with:

```bash
cd /mnt/d/Proga/Game_of_life
/mnt/d/Anaconda3/NewAnaconda/python.exe -m Real_game_of_life.GNN.evaluate_rollout \
  --cache Real_game_of_life/GNN/cache/frame_graphs_dynamic_v2_full_balanced.pt \
  --checkpoints \
    Real_game_of_life/GNN/runs/gnn_baseline_local_full_balanced/best.pt \
    Real_game_of_life/GNN/runs/gnn_rollout_bptt3_full_seed17/best.pt \
  --labels baseline,bptt3 \
  --horizons 1,3,5,10,20 \
  --out Real_game_of_life/GNN/runs/rollout_eval_baseline_vs_bptt3.json \
  --device cpu
```

The evaluator writes JSON and CSV reports with position mean/median/RMSE,
`shape_r_norm_*` RMSE, matched-node counts, and valid predicted shape-profile
fractions for each horizon.

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
`shape_rmse`. Division horizon metrics (`division_h3_*`, `division_h5_*`,
`division_h10_*`) are usually more informative than the one-step division head.
