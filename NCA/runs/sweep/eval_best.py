from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from NCA.dataset import build_dataloaders
from NCA.model import NCA
from NCA.train import stochastic_eval
from NCA.utils import build_initial_state, compute_binary_mask_metrics, load_checkpoint, rollout_model, set_seed, visible_to_probability
from NCA.visualize import plot_metric_curves, plot_triptych, plot_uncertainty_heatmap, save_rollout_animation

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
    'eval_threshold': 0.45,
    'num_rollouts': 8,
    'use_alive_mask': False,
    'seed': 0,
}
run_dir = PROJECT_ROOT / 'NCA' / 'runs' / 'sweep'
best_dir = run_dir / 'pos_weight_2'
out_dir = run_dir / 'best_visuals'
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

stoch = stochastic_eval(
    model=model,
    loader=loaders['val_rollout'],
    device=torch.device('cpu'),
    num_rollouts=CONFIG['num_rollouts'],
    data_channels=CONFIG['data_channels'],
    hidden_channels=CONFIG['hidden_channels'],
    primary_channel=CONFIG['primary_channel'],
    supervised_loss=CONFIG['supervised_loss'],
    threshold=CONFIG['eval_threshold'],
)

batch = next(iter(loaders['val_rollout']))
state0 = build_initial_state(batch['input_visible'], hidden_channels=CONFIG['hidden_channels'], hidden_init='zeros')
steps = int(batch['horizons'].max().item())
with torch.no_grad():
    det_rollout = rollout_model(model, state0, steps=steps, stochastic=False)
    stoch_rollouts = torch.stack([rollout_model(model, state0, steps=steps, stochastic=True) for _ in range(CONFIG['num_rollouts'])], dim=0)

prediction_prob = visible_to_probability(det_rollout[-1, 0, :CONFIG['data_channels']].detach().cpu(), supervised_loss=CONFIG['supervised_loss'])
primary_slice = slice(CONFIG['primary_channel'], CONFIG['primary_channel'] + 1)
prob_primary = prediction_prob[primary_slice]
target_primary = batch['targets_visible'][0, -1, primary_slice].detach().cpu()
thresholded = (prob_primary >= CONFIG['eval_threshold']).float()
sample_metrics = compute_binary_mask_metrics(prob_primary.unsqueeze(0), target_primary.unsqueeze(0), threshold=CONFIG['eval_threshold'])

plot_triptych(batch['input_visible'][0], batch['targets_visible'][0, -1], prediction_prob, out_dir / 'triptych.png', title='Best raw probability', channel_index=CONFIG['primary_channel'])
plot_triptych(batch['input_visible'][0], target_primary, thresholded, out_dir / 'triptych_thresholded.png', title=f"Best thresholded @ {CONFIG['eval_threshold']}", channel_index=0)
plot_uncertainty_heatmap(stoch_rollouts.detach().cpu(), out_dir / 'uncertainty.png', visible_channel=CONFIG['primary_channel'], step_index=-1)
save_rollout_animation(visible_to_probability(det_rollout[:, 0, :CONFIG['data_channels']].detach().cpu(), supervised_loss=CONFIG['supervised_loss']), out_dir / 'det_rollout.gif', channel_index=CONFIG['primary_channel'])

summary = {
    'best_checkpoint': str(best_dir / 'checkpoint_latest.pt'),
    'threshold': CONFIG['eval_threshold'],
    'stochastic_eval': stoch,
    'sample_binary_metrics': sample_metrics,
    'artifacts': {
        'triptych': str(out_dir / 'triptych.png'),
        'thresholded': str(out_dir / 'triptych_thresholded.png'),
        'uncertainty': str(out_dir / 'uncertainty.png'),
        'rollout_gif': str(out_dir / 'det_rollout.gif'),
    },
}
(out_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
