"""Plot the published, verified development summary; no model or environment calls.

Reproduce with matplotlib==3.10.9 and numpy==2.2.6:
  python protected-boundary-development-v1.py --data path/to/robomme-protected-boundary.json
"""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[2] / 'data/robomme-protected-boundary.json')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    payload = args.data.read_bytes()
    data = json.loads(payload)
    assert data['readout']['completed_at_observation'] == 56
    groups = data['development_readout']['groups']
    old = data['protected_old8_readout']['versions']
    models = ['released', 'full_correction_seed7', 'protected_seed7']
    labels = ['Released', 'Correction BC', 'Protected action-expert BC']
    colors = ['#7B8794', '#C9754E', '#356BA6']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12,
                         'axes.labelsize': 10, 'axes.edgecolor': '#B5BDC5', 'axes.linewidth': .7,
                         'text.color': '#172532', 'axes.labelcolor': '#263746', 'xtick.color': '#334656',
                         'ytick.color': '#53616F', 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'svg.hashsalt': 'robomme-protected-boundary-v1'})
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.25), gridspec_kw={'width_ratios': [1.7, 1]})
    fig.subplots_adjust(left=.065, right=.985, bottom=.22, top=.73, wspace=.22)
    fig.suptitle('Protected action-expert BC: complete development readout', x=.065, y=.97, ha='left', fontsize=16, weight='semibold')
    fig.text(.065, .90, 'Terminal success on fixed cases; trained models use final-10k deployment; no extra expert queries', fontsize=10.5, color='#53616F')
    tasks = ['PatternLock', 'PickHighlight', 'Overall']
    xs = np.arange(3)
    for j, (key, label, color) in enumerate(zip(models, labels, colors)):
        g = groups[key + ':fresh16']
        counts = [g['tasks'][t] for t in tasks[:2]] + [g]
        ys = [100 * c['successes'] / c['cases'] for c in counts]
        bars = axes[0].bar(xs + (j - 1) * .235, ys, width=.215, color=color, label=label, zorder=3)
        for b, c in zip(bars, counts):
            axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 3.2, f"{c['successes']}/{c['cases']}", ha='center', va='bottom', fontsize=9, color='#263746')
    axes[0].set_xticks(xs, tasks)
    axes[0].set_title('A   New development instances', loc='left', pad=16, weight='semibold')
    axes[0].set_ylabel('Terminal success (%)')
    axes[0].legend(frameon=False, loc='lower left', bbox_to_anchor=(-.012, 1.23), ncol=3, fontsize=9, handlelength=1.2, columnspacing=1.2)
    old_keys = ['initial', 'correction_only', 'mixture', 'pure_replay', 'protected']
    old_labels = ['Released', 'Correction\nBC', 'Mixture', 'Pure\nreplay', 'Protected']
    old_colors = [colors[0], colors[1], '#C5AC7D', '#7EA393', colors[2]]
    vals = [old[k] for k in old_keys]
    bars = axes[1].bar(np.arange(5), [100*v['success']/v['total'] for v in vals], width=.65, color=old_colors, zorder=3)
    for b, v in zip(bars, vals):
        axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+3.2, f"{v['success']}/{v['total']}", ha='center', va='bottom', fontsize=9)
    axes[1].set_xticks(np.arange(5), old_labels, fontsize=9)
    axes[1].set_title('B   Previously used development cases', loc='left', pad=16, weight='semibold')
    for ax in axes:
        ax.set_ylim(0, 113)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis='y', color='#E5E9ED', linewidth=.7, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='both', length=0, pad=7)
        ax.set_axisbelow(True)
    fig.text(.065, .105, 'One training seed. New block: 16 managed development instances, evaluated for all three models.', fontsize=9, color='#53616F')
    fig.text(.065, .065, 'Old block: 8 cases reused for development; prior results cited without reruns. No confirmation, root-cause, or RSI claim.', fontsize=9, color='#53616F')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / 'protected-boundary-development-v1'
    description = 'Data SHA256: ' + hashlib.sha256(payload).hexdigest() + '; owner report SHA256: ' + data['development_readout']['owner_report_sha256']
    fig.savefig(stem.with_suffix('.svg'), metadata={'Date': None, 'Description': description})
    svg_path = stem.with_suffix('.svg')
    svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines()) + '\n')
    fig.savefig(stem.with_suffix('.pdf'), metadata={'CreationDate': None, 'ModDate': None, 'Subject': description})
    fig.savefig(stem.with_suffix('.png'), dpi=190, metadata={'Description': description})
    plt.close(fig)
    print(json.dumps({'figure': stem.name, 'data_sha256': hashlib.sha256(payload).hexdigest(), 'matplotlib': matplotlib.__version__, 'numpy': np.__version__}))


if __name__ == '__main__':
    main()
