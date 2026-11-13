"""Render saved full-window qualification evidence without running the model."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np


def log_ticks(ax):
    lo, hi = ax.get_ylim()
    ax.yaxis.set_major_locator(FixedLocator(np.geomspace(lo, hi, 6)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.5f}'.rstrip('0').rstrip('.')))
    ax.yaxis.set_minor_formatter(NullFormatter())


def reports(root, stop):
    items = [json.loads(p.read_text()) for p in sorted((root/'evaluation').glob('update_*/report.json'))]
    return [r for r in items if r['accepted_updates'] <= stop]


def energy_panel(root, update, cases, out, comparison=None):
    folder = root/'evaluation'/f'update_{update:08d}'
    fig, axes = plt.subplots(7, 3, figsize=(15, 18), layout='constrained')
    for ax, c in zip(axes.flat, cases):
        with np.load(folder/f"{c['case_id']}.npz", allow_pickle=False) as z:
            t, e, ref = z['times'], z['model_energy'], z['teacher_energy']
            fail = float(z['failure_t'])
        ax.semilogy(t, ref, color='#2463eb', lw=1.1, label='Kinetic reference')
        if comparison:
            prior = comparison/'evaluation'/f'update_{update:08d}'/f"{c['case_id']}.npz"
            if prior.exists():
                with np.load(prior, allow_pickle=False) as z:
                    ax.semilogy(z['times'][:len(z['model_energy'])], z['model_energy'],
                                color='#d3913f', lw=.9, alpha=.8, label='Previous 5 + 10 training')
        ax.semilogy(t[:len(e)], e, color='#743b9d', lw=1, label='Full 15 training')
        if np.isfinite(fail):
            ax.axvline(fail, color='#b91c1c', ls=':', lw=1)
            ax.text(.02, .04, f'Nonfinite at t={fail:.3f}', transform=ax.transAxes,
                    color='#b91c1c', fontsize=8)
        regime = c['regime'].replace('nonlinear_landau_', '').replace('linear_landau', 'linear')
        ax.set_title(f"k₀={c['fundamental']:.2f} | {regime} | {c['family']}", fontsize=9)
        ax.set_xlim(0, 120); ax.set_xticks([0, 40, 80, 120])
        ax.grid(alpha=.18); ax.tick_params(labelsize=8)
    axes[0, 0].legend(fontsize=7, frameon=False)
    for ax in axes[-1]: ax.set_xlabel('t')
    for ax in axes[:, 0]: ax.set_ylabel('Electric energy')
    fig.suptitle(f'All 21 held-out autonomous energy trajectories | {update} accepted updates')
    fig.savefig(out/f'autonomous_energy_update_{update:08d}.png', dpi=170)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--updates', type=int, required=True)
    p.add_argument('--comparison', type=Path, required=True)
    args = p.parse_args(); root = args.root
    out = root/'analysis'/f'update_{args.updates:08d}'
    out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((root/'protocol.json').read_text())
    count = protocol['updates_per_epoch']
    with np.load(root/'training'/f'state_{args.updates:08d}.npz', allow_pickle=False) as z:
        rows = z['loss_rows']; record = json.loads(str(z['record']))
    rr = reports(root, args.updates); prior = reports(args.comparison, args.updates)
    vals = [(r['accepted_updates'], r['validation_losses']['full15_l2']) for r in rr
            if r['validation_losses']['full15_l2'] is not None]
    formula = r'$\mathcal{L}_{15}=\frac{1}{B}\sum_i\frac{\|\widehat{\delta M}_i-\delta M_i\|_{[a_i,a_i+15]}}{\|\delta M_i\|_{[a_i,a_i+15]}}$'
    if len(rows):
        fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
        ax.plot(rows[:, 0], rows[:, 1], color='#111827', lw=1, label='Training batch loss')
        ax.plot([v[0] for v in vals], [v[1] for v in vals], '-o', color='#743b9d', label='Validation loss')
        ax.set_xlabel('Accepted updates'); ax.set_ylabel(formula)
        ax.set_yscale('log'); log_ticks(ax); ax.grid(alpha=.2); ax.legend(frameon=False)
        ax.set_title('Full 15-unit objective: training and validation')
        fig.savefig(out/'training_validation_updates.png', dpi=180); plt.close(fig)
    # Each batch mean is weighted by its number of windows, including the final short batch.
    manifest = json.loads((root/'manifest.json').read_text())
    # Batch ordering belongs to this run's preserved source, including its old module paths.
    sys.path.insert(0, str(root/'frozen_source'))
    from model.train.wavenumber_data import epoch_batches
    batch_sizes = np.asarray([len(b) for b in epoch_batches(manifest, 1, protocol['config']['seed'])])
    assert len(batch_sizes) == count and batch_sizes.sum() == protocol['windows_per_epoch']
    epochs = args.updates//count
    train = [float(np.average(rows[e*count:(e+1)*count, 1], weights=batch_sizes)) for e in range(epochs)]
    if epochs:
        vv = [(u/count, v) for u, v in vals if u % count == 0]
        fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
        ax.plot(np.arange(1, epochs+1), train, '-o', color='#111827', label='Training loss')
        ax.plot([v[0] for v in vv], [v[1] for v in vv], '-o', color='#743b9d', label='Validation loss')
        ax.set_xlabel('Epoch'); ax.set_ylabel(formula); ax.set_xticks(np.arange(epochs+1))
        ax.set_yscale('log'); log_ticks(ax); ax.grid(alpha=.2); ax.legend(frameon=False)
        fig.savefig(out/'complete_epoch_train_validation.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
    for items, label, color, key in [(prior, 'Previous 5 + 10 training', '#d3913f', 'window_rms_l2'),
                                    (rr, 'Full 15 training', '#743b9d', 'legacy_scored10_l2')]:
        for ax, metric in zip(axes, ['legacy', 'mean_epsilon_E', 'mean_log_energy_rmse']):
            values = [(r['accepted_updates'], r['validation_losses'].get(key) if metric == 'legacy' else r[metric]) for r in items]
            values = [(u, v) for u, v in values if v is not None and np.isfinite(v)]
            ax.plot([v[0] for v in values], [v[1] for v in values], '-o', color=color, label=label, ms=4)
            ax.set_xlabel('Accepted updates'); ax.set_yscale('log'); ax.grid(alpha=.2)
    for ax, title in zip(axes, ['Same legacy ten-unit validation', 'Autonomous mean field error', 'Autonomous log-energy RMSE']):
        ax.set_title(title, fontsize=11)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(out/'comparable_metrics.png', dpi=180); plt.close(fig)
    cases = [c for c in manifest['cases'] if c['split'] == 'heldout']
    finite = [r for r in rr if r['nonfinite_count'] == 0]
    selections = {'last': args.updates}
    if finite:
        selections['lowest_field_error'] = min(finite, key=lambda r:r['mean_epsilon_E'])['accepted_updates']
        selections['lowest_log_energy_error'] = min(finite, key=lambda r:r['mean_log_energy_rmse'])['accepted_updates']
    for u in sorted(set(selections.values())):
        if (root/'evaluation'/f'update_{u:08d}'/'report.json').exists():
            energy_panel(root, u, cases, out, args.comparison)
    summary = dict(record=record, complete_epochs=epochs, complete_epoch_train_losses=train,
        same_objective_validation=vals, selections=selections,
        evaluations=[{k:r[k] for k in ['accepted_updates','finite_count','nonfinite_count','mean_epsilon_E','mean_log_energy_rmse','validation_losses']} for r in rr],
        partial_epoch=bool(args.updates % count),
        note='No EMA. Training uses changing weights. Only complete sweeps are called epochs. Legacy validation is plotted separately because the objective differs.')
    (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2))
    print(out)


if __name__ == '__main__':
    main()
