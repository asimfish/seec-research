#!/usr/bin/env python3
"""Descriptive 2x2 outcomes from verified artifacts; no selection or tuning."""
import argparse
import json
from pathlib import Path

import numpy as np

ARMS=['offline_action','offline_facts','student_action','student_facts']
NAMES=['Offline + actions','Offline + facts','Student + actions','Student + facts']


def analyze(folder):
    def sha(p):
        import hashlib
        return hashlib.sha256(p.read_bytes()).hexdigest()
    s=json.loads((folder/'summary.json').read_text());v=json.loads((folder/'verification.json').read_text());p=json.loads((folder/'plan.json').read_text())
    assert v['status']=='PASS' and v['summary_sha256']==sha(folder/'summary.json')
    rows={}
    for arm in ARMS:
        raw=s['arms'][arm]
        rows[arm]={'mean':float(np.mean([r['success_rate'] for r in raw])),
                   'by_seed':[r['success_rate'] for r in raw],
                   'by_trial':[float(np.mean([r['by_trial'][i]['success_rate'] for r in raw])) for i in range(5)],
                   'mean_cost_steps':float(np.mean([r['mean_cost_steps'] for r in raw])),
                   'reset_mean':float(np.mean([s['evaluations'][f'{arm}/{seed}/reset_each_episode']['success_rate'] for seed in p['training_seeds']])),
                   'collection_actions':sum(r['collection_steps'] for r in s['collection'] if r['arm']==arm),
                   'sequence_counts_per_seed':[1239,1416],
                   'action_targets_final_per_seed':[next(r['rounds'][-1]['action_targets'] for r in s['models'] if r['arm']==arm and r['seed']==seed) for seed in p['training_seeds']]}
    contrasts={}
    for key,a,b in [('coverage_action','student_action','offline_action'),('coverage_facts','student_facts','offline_facts'),
                    ('facts_offline','offline_facts','offline_action'),('facts_student','student_facts','student_action')]:
        differences=[x-y for x,y in zip(rows[a]['by_seed'],rows[b]['by_seed'])]
        contrasts[key]={'mean':float(np.mean(differences)),'by_seed':differences,'all_seeds_positive':all(x>0 for x in differences)}
    interaction=[x-y for x,y in zip(contrasts['facts_student']['by_seed'],contrasts['facts_offline']['by_seed'])]
    checks={key:contrasts[key]['mean']>=.05 and contrasts[key]['all_seeds_positive'] for key in ['coverage_action','facts_student']}
    r={'schema':'student-coverage-analysis-v1','summary_sha256':sha(folder/'summary.json'),'verification_sha256':sha(folder/'verification.json'),
       'plan_id':p['plan_id'],'seeds':p['training_seeds'],'arms':rows,'contrasts':contrasts,
       'interaction':{'mean':float(np.mean(interaction)),'by_seed':interaction},'registered_checks':checks,
       'stop_reason':'registered-12-run-budget-completed','limits':['Descriptive three-seed reused-panel diagnostic, not confirmatory inference',
        'Common post-parent action-label validity guard; no guard ablation in this panel',
        'Same initial weights, updates and sequence counts, not equal label-token or collection-step cost',
        'DAgger-style baseline is prior art, not a new OPD algorithm']}
    (folder/'analysis.json').write_text(json.dumps(r,indent=2)+'\n')
    lines=['# Student histories × factual learning: verified result','',
           'Four arms × three paired seeds, all initialized from the same action-only parent. Two collect/update rounds, 600 additional updates each. Teacher-free final evaluation; no checkpoint selection.','',
           '| Arm | Mean success | Seeds 7 / 19 / 41 | Memory reset | New collection actions |','|---|---:|---|---:|---:|']
    for arm,name in zip(ARMS,NAMES):
        row=rows[arm];lines.append(f"| {name} | {row['mean']:.2%} | {' / '.join(f'{x:.2%}' for x in row['by_seed'])} | {row['reset_mean']:.2%} | {row['collection_actions']:,} |")
    lines+=['','| Paired contrast | Mean difference (percentage points) | Seeds (percentage points) |','|---|---:|---|']
    for key,row in contrasts.items():lines.append(f"| {key} | {100*row['mean']:+.2f} | {' / '.join(f'{100*x:+.2f}' for x in row['by_seed'])} |")
    lines+=['',f"Interaction (facts gain under student collection minus offline): {100*r['interaction']['mean']:+.2f} points.",
             f"Registered coverage gain check: {checks['coverage_action']}; facts survive coverage: {checks['facts_student']}.",
             '', 'Both checks require at least five percentage points in the seed mean and positive difference in all three seeds. A failed prediction is retained, not tuned away.',
             '', 'All arms apply the common observable guard: once bowl contents already contradict the goal, ignore the action-positive loss while retaining the actual history and factual targets. Parent metrics are untouched. This is not an isolated estimate of the guard benefit.',
             '', 'The base source contains 1,062 histories; round pools have 1,239 and 1,416 per seed. Optimizer steps, sequence batch size and batch indices match. Actual collection actions and valid labels differ and are reported. No equal-token or budget-advantage claim is made.',
             '', f"Coverage: {v['evaluation_records']:,} evaluation records, {v['source_sequences']:,} new histories. Independently replayed {v['source_actions']:,} collected actions and {v['evaluation_actions']:,} evaluation actions; all raw differences zero. Student collector actions and all final outputs recomputed from frozen checkpoint bytes.",
             '', 'This reuses the prior panel and training seeds; three paired seeds are descriptive. It is a DAgger-style comparator, not original GMPS, DREAM, VariBAD or OPD reproduction. Real robotics, execution uncertainty and the future-adaptation selector remain untested.']
    (folder.parent/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return r


def plot(r,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    destination.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.3),gridspec_kw={'width_ratios':[1.5,1]})
    colors=['#8b97ae','#6386c9','#238e9a','#285bd6']
    for i,arm in enumerate(ARMS):
        row=r['arms'][arm];axes[0].bar(i,row['mean'],width=.6,color=colors[i],alpha=.85)
        axes[0].scatter([i-.12,i,i+.12],row['by_seed'],s=20,c='#18243a',zorder=3)
        axes[0].text(i,max(row['by_seed'])+.03,f"{row['mean']:.1%}",ha='center',weight='bold')
    axes[0].set_xticks(range(4),['Offline\naction','Offline\nfacts','Student\naction','Student\nfacts'])
    axes[0].set_ylabel('Success averaged over five attempts');axes[0].set_ylim(0,1.12)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1));axes[0].set_title('A  Collection × learning objective',loc='left')
    for i,key in enumerate(['facts_offline','facts_student']):
        vals=np.asarray(r['contrasts'][key]['by_seed'])*100
        axes[1].scatter([i-.1,i,i+.1],vals,color=colors[1+i*2],s=40,zorder=3)
        axes[1].plot([i-.22,i+.22],[vals.mean()]*2,color='#18243a',lw=2)
        axes[1].text(i,vals.max()+2.5 if i==0 else vals.mean()-4,f"{vals.mean():+.1f} pp",ha='center',weight='bold')
    axes[1].axhline(0,color='#9aa4b4',linestyle='--',lw=1)
    axes[1].set_xticks([0,1],['Offline histories','Student histories']);axes[1].set_xlim(-.5,1.5)
    vals=[100*x for k in ['facts_offline','facts_student'] for x in r['contrasts'][k]['by_seed']]
    axes[1].set_ylim(min(-5,min(vals)-6),max(10,max(vals)+12))
    axes[1].set_ylabel('Factual auxiliary gain (percentage points)');axes[1].set_title('B  Paired gain, all three seeds',loc='left')
    for ax in axes:ax.grid(axis='y',color='#e6eaf0',lw=.7);ax.set_axisbelow(True)
    fig.text(.065,.015,'Same initial checkpoint and 600 extra updates · 3 paired seeds · dots are seeds, not confidence intervals',fontsize=8.3,color='#526079')
    fig.tight_layout(rect=(0,.05,1,1))
    for suffix in ['svg','png','pdf']:fig.savefig(destination/f'cooking_student_coverage.{suffix}',dpi=180,bbox_inches='tight')
    # A separate print figure uses the PDF core font set; the public SVG and
    # primary plot keep their native text. This avoids outlining/clipping
    # artifacts while allowing embedded Type-1 fonts in the paper.
    from matplotlib.text import Text
    for artist in fig.findobj(Text):artist.set_fontfamily('Helvetica')
    with plt.rc_context({'pdf.use14corefonts':True,'axes.unicode_minus':False}):
        fig.savefig(destination/'cooking_student_coverage_core.pdf',bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',type=Path,required=True);p.add_argument('--no-plot',action='store_true')
    args=p.parse_args();r=analyze(args.folder)
    if not args.no_plot:plot(r,args.folder.parent/'figures')
    print(json.dumps(r,indent=2))
