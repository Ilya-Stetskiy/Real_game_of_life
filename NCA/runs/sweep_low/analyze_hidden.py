from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from NCA.dataset import build_dataloaders
from NCA.model import NCA
from NCA.utils import build_initial_state, load_checkpoint, rollout_model, set_seed, visible_to_probability

CONFIG = {
    'data_root': PROJECT_ROOT / 'NCA' / 'data',
    'pattern': '*.npy',
    'split_mode': 'by_file',
    'split_ratios': (0.6, 0.2, 0.2),
    'train_steps': (8, 16),
    'eval_steps': {'one_step': 1, 'rollout': 16, 'stochastic': 16},
    'batch_size': 8,
    'kernel_size': 3,
    'model_width': 64,
    'update_prob': 0.5,
    'data_channels': 2,
    'hidden_channels': 8,
    'primary_channel': 0,
    'supervised_loss': 'bce_dice',
    'eval_threshold': 0.30,
    'use_alive_mask': False,
    'seed': 0,
}
run_dir = PROJECT_ROOT / 'NCA' / 'runs' / 'sweep_low'
best_dir = run_dir / 'pos_weight_0.25'
out_dir = run_dir / 'hidden_analysis'
out_dir.mkdir(parents=True, exist_ok=True)
set_seed(CONFIG['seed'])

loaders, _ = build_dataloaders(
    data_root=CONFIG['data_root'], pattern=CONFIG['pattern'], split_mode=CONFIG['split_mode'], split_ratios=CONFIG['split_ratios'],
    train_steps=CONFIG['train_steps'], eval_steps=CONFIG['eval_steps'], batch_size=CONFIG['batch_size'], eval_batch_size=CONFIG['batch_size'],
    seed=CONFIG['seed'], num_workers=0, data_channels=CONFIG['data_channels'], primary_channel=CONFIG['primary_channel'],
)
model = NCA(
    state_channels=CONFIG['data_channels'] + CONFIG['hidden_channels'], model_width=CONFIG['model_width'], kernel_size=CONFIG['kernel_size'],
    update_prob=CONFIG['update_prob'], use_alive_mask=CONFIG['use_alive_mask'], primary_channel=CONFIG['primary_channel'],
)
load_checkpoint(best_dir / 'checkpoint_latest.pt', model=model, optimizer=None, map_location='cpu')
model.eval()

batch = next(iter(loaders['val_rollout']))
state0 = build_initial_state(batch['input_visible'], hidden_channels=CONFIG['hidden_channels'], hidden_init='zeros')
steps = int(batch['horizons'].max().item())
with torch.no_grad():
    rollout = rollout_model(model, state0, steps=steps, stochastic=False)

final_state = rollout[-1]
hidden = final_state[:, CONFIG['data_channels']:]
observed_logits = final_state[:, :CONFIG['data_channels']]
observed_prob = visible_to_probability(observed_logits, supervised_loss=CONFIG['supervised_loss'])
target = batch['targets_visible'][:, -1]
primary_prob = observed_prob[:, CONFIG['primary_channel']]
primary_target = target[:, CONFIG['primary_channel']]

hidden_np = hidden.detach().cpu().numpy()
summary = []
flat_target = primary_target.detach().cpu().numpy().reshape(-1)
flat_prob = primary_prob.detach().cpu().numpy().reshape(-1)
for ch in range(CONFIG['hidden_channels']):
    values = hidden_np[:, ch]
    flat = values.reshape(-1)
    if np.std(flat) < 1e-8:
        corr_target = 0.0
        corr_prob = 0.0
    else:
        corr_target = float(np.corrcoef(flat, flat_target)[0, 1])
        corr_prob = float(np.corrcoef(flat, flat_prob)[0, 1])
    temporal = rollout[:, :, CONFIG['data_channels'] + ch].detach().cpu().numpy()
    step_mean_abs = np.mean(np.abs(temporal), axis=(1, 2, 3))
    summary.append({
        'hidden_channel': ch,
        'mean': float(values.mean()),
        'std': float(values.std()),
        'mean_abs': float(np.abs(values).mean()),
        'min': float(values.min()),
        'max': float(values.max()),
        'corr_with_target_primary': corr_target,
        'corr_with_prediction_primary': corr_prob,
        'mean_abs_step0': float(step_mean_abs[0]),
        'mean_abs_step_final': float(step_mean_abs[-1]),
        'mean_abs_temporal_delta': float(step_mean_abs[-1] - step_mean_abs[0]),
    })

# Heatmap montage for first sample: input, target, prediction, hidden channels.
fig, axes = plt.subplots(3, 4, figsize=(14, 10), constrained_layout=True)
axes = axes.ravel()
images = [
    ('input primary', batch['input_visible'][0, CONFIG['primary_channel']].detach().cpu().numpy()),
    ('target primary', primary_target[0].detach().cpu().numpy()),
    ('pred primary', primary_prob[0].detach().cpu().numpy()),
    ('pred mask', (primary_prob[0].detach().cpu().numpy() >= CONFIG['eval_threshold']).astype(np.float32)),
]
for i, (title, img) in enumerate(images):
    axes[i].imshow(img, cmap='viridis')
    axes[i].set_title(title)
    axes[i].axis('off')
for ch in range(CONFIG['hidden_channels']):
    ax = axes[4 + ch]
    img = hidden_np[0, ch]
    vmax = max(abs(float(img.min())), abs(float(img.max())), 1e-6)
    ax.imshow(img, cmap='coolwarm', vmin=-vmax, vmax=vmax)
    ax.set_title(f'hidden {ch}')
    ax.axis('off')
fig.savefig(out_dir / 'hidden_channels_montage.png', dpi=150)
plt.close(fig)

# Temporal energy plot.
fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
for ch in range(CONFIG['hidden_channels']):
    temporal = rollout[:, :, CONFIG['data_channels'] + ch].detach().cpu().numpy()
    step_mean_abs = np.mean(np.abs(temporal), axis=(1, 2, 3))
    ax.plot(np.arange(1, steps + 1), step_mean_abs, label=f'h{ch}')
ax.set_xlabel('rollout step')
ax.set_ylabel('mean abs hidden value')
ax.legend(ncol=4, fontsize=8)
ax.grid(True, alpha=0.3)
fig.savefig(out_dir / 'hidden_temporal_energy.png', dpi=150)
plt.close(fig)

result = {
    'checkpoint': str(best_dir / 'checkpoint_latest.pt'),
    'summary': summary,
    'artifacts': {
        'hidden_channels_montage': str(out_dir / 'hidden_channels_montage.png'),
        'hidden_temporal_energy': str(out_dir / 'hidden_temporal_energy.png'),
    },
}
(out_dir / 'hidden_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
