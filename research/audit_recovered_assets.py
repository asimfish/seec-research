"""Read-only audit of historical scene media and checkpoint semantics.

Never starts Isaac, modifies the source archive, or equates file replay with
physics continuation. The mock probes exercise the archived adapter's setters.
"""
import argparse
import copy
import gzip
import hashlib
import importlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class SetterProbe:
    def __init__(self, names=()):
        self.joint_names = names
        self.calls = {}

    def __getattr__(self, name):
        if name.startswith(('write_', 'set_')):
            def record(value, *args, **kwargs):
                self.calls[name] = value.detach().cpu().clone()
            return record
        raise AttributeError(name)


def probe_adapter(schema, runtime, original):
    cp = copy.deepcopy(original)
    # Use a nonzero velocity input accepted by the actual stored schema. This
    # is a contract counterexample, not an executed simulator trajectory.
    for row in cp['physics']['bodies']:
        row['root_state'][7] = .25
    for row in cp['physics']['articulations']:
        row['root_state'][7] = .25
        row['joint_velocity'] = [.25] * len(row['joint_names'])
        row['joint_velocity_target'] = [.125] * len(row['joint_names'])
    scene = SimpleNamespace(num_envs=1,
        rigid_objects={x['id']: SetterProbe() for x in cp['physics']['bodies']},
        articulations={x['id']: SetterProbe(x['joint_names']) for x in cp['physics']['articulations']})
    ns = {'scene': scene, 'PLAN': {}, 'dev': 'cpu', 'CHECKPOINT_IDENTITY': cp['identity']}
    rec = SimpleNamespace(arm=None)
    runtime.restore(ns, rec, cp)
    art = next(iter(scene.articulations.values()))
    written_qd = art.calls['write_joint_state_to_sim'] if False else None
    # The recorder needs both positional arguments for the joint-state API;
    # capture these separately below rather than deriving behavior from names.
    row = cp['physics']['articulations'][0]
    captured = {}
    def joint_state(q, qd, **kwargs):
        captured['qd'] = qd.detach().cpu().clone()
    scene.articulations[row['id']].write_joint_state_to_sim = joint_state
    runtime.restore(ns, rec, cp)
    root_written = scene.articulations[row['id']].calls['write_root_velocity_to_sim']
    velocity_written = captured['qd']
    before = sum(len(x.calls) for x in list(scene.rigid_objects.values()) + list(scene.articulations.values()))
    bad_ns = dict(ns, CHECKPOINT_IDENTITY={**cp['identity'], 'scene_id': 'wf_other'})
    try:
        runtime.restore(bad_ns, rec, cp)
        identity_rejected = False
    except RuntimeError:
        identity_rejected = True
    after = sum(len(x.calls) for x in list(scene.rigid_objects.values()) + list(scene.articulations.values()))
    missing_ns = dict(ns); missing_ns.pop('CHECKPOINT_IDENTITY')
    try:
        runtime.restore(missing_ns, rec, cp)
        missing_rejected = False
    except RuntimeError:
        missing_rejected = True
    return {'scope': 'CPU setter contract probe, not simulator execution',
        'schema_accepts_nonzero_velocities': bool(schema.validate_checkpoint(cp)),
        'saved_joint_velocity_rad_s': .25,
        'restored_joint_velocity_max_rad_s': float(velocity_written.abs().max()),
        'saved_root_velocity_m_s': .25,
        'restored_articulation_root_velocity_max': float(root_written.abs().max()),
        'preserves_nonzero_joint_velocity': bool(torch.all(velocity_written == .25)),
        'identity_mismatch_rejected_before_writes': identity_rejected and before == after,
        'missing_identity_rejected': missing_rejected,
        'task_graph_position_restored': hasattr(rec, 'task_graph_position'),
        'rng_state_restored': 'rng' in ns}


def video_info(path):
    data = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,nb_frames,r_frame_rate,duration', '-of', 'json', str(path)], text=True))
    return {**data['streams'][0], 'sha256': sha(path), 'bytes': path.stat().st_size}


def audit_episode(folder, renders):
    meta = json.loads((folder / 'meta.json').read_text())
    rows, negative_rows, protected_negative_rows, deep_rows = 0, 0, 0, 0
    min_separation = 0.; last_negative_tick = -1; ticks = []
    with gzip.open(folder / 'contacts.jsonl.gz', 'rt') as stream:
        for line in stream:
            entry = json.loads(line)
            if entry.get('type') != 'tick':
                continue
            tick = int(entry['tick']); ticks.append(tick); rows += 1
            negative = protected = deep = False
            for contact in entry.get('contacts', []):
                sep = contact.get('separation_m')
                if sep is None:
                    continue
                sep = float(sep); min_separation = min(min_separation, sep)
                if sep < 0:
                    negative = True; last_negative_tick = max(last_negative_tick, tick)
                    names = [str(contact.get(k, '')) for k in ('actor0','actor1')]
                    protected |= any('/World/Robots/' in s or '/World/StationA/precursor_' in s for s in names)
                deep |= sep < -.003
            negative_rows += negative; protected_negative_rows += protected; deep_rows += deep
    with np.load(folder / 'steps.npz', allow_pickle=False) as arrays:
        step_count = len(arrays['t'])
    videos = {p.stem: video_info(p) for p in sorted((renders / 'videos').glob('*.mp4'))}
    return {'run_id': meta['run_id'], 'task_success_recorded': meta['metrics'].get('task_success'),
        'grasp_mode': meta.get('grasp_mode'), 'physics_hz': round(1/meta['dt_s']),
        'source_ticks': meta['ticks'], 'step_rows': step_count, 'contact_tick_rows': rows,
        'contact_ticks_unique': len(ticks) == len(set(ticks)),
        'contact_ticks_contiguous': ticks == list(range(len(ticks))),
        'any_negative_separation_tick_rows': negative_rows,
        'robot_or_bottle_negative_separation_tick_rows': protected_negative_rows,
        'deeper_than_3mm_tick_rows': deep_rows, 'minimum_separation_m': min_separation,
        'last_negative_time_s': last_negative_tick * meta['dt_s'],
        'placements_mm': meta['metrics'].get('placements_mm'),
        'full_views': videos, 'checkpoint_files_in_episode': len(list(folder.glob('*checkpoint*'))),
        'source_sha256': {name: sha(folder/name) for name in ['meta.json','steps.npz','contacts.jsonl.gz']},
        'scope': 'Persisted physical demonstration; not whole-scene zero penetration, formal benchmark qualification or fresh physics replay'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-source', type=Path, required=True)
    p.add_argument('--checkpoint-run', type=Path, required=True)
    p.add_argument('--bottle-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    # Snapshot only the trusted archived modules locally; never alter them.
    import shutil
    with tempfile.TemporaryDirectory(prefix='scene-checkpoint-readonly-') as work:
        for name in ['checkpoint_schema.py','checkpoint_runtime.py']:
            shutil.copyfile(a.checkpoint_source/name, Path(work)/name)
        sys.path.insert(0, work)
        schema = importlib.import_module('checkpoint_schema')
        runtime = importlib.import_module('checkpoint_runtime')
        cp = schema.load_checkpoint(a.checkpoint_run/'carry_checkpoint.npz')
        target = Path(work)/'roundtrip.npz'; schema.save_checkpoint(target, cp)
        cp2 = schema.load_checkpoint(target)
        serialization = schema.canonical_json(cp) == schema.canonical_json(cp2)
        probe = probe_adapter(schema, runtime, cp)
    historical = json.loads((a.checkpoint_run/'independent_audit.json').read_text())
    result = {'schema': 'scene-recovery-audit-v1', 'overall_stage_resume': 'PARTIAL_NOT_COMPLETE',
        'fresh_serialization_roundtrip_exact': serialization, 'fresh_adapter_probe': probe,
        'historical_carry': {k: historical[k] for k in ['status','scope','placement_error_m','settled_relative_pose_difference','whole_scene_resume_gate']},
        'carry_mode': cp['carry_mode'], 'checkpoint_body_count': len(cp['physics']['bodies']),
        'checkpoint_randomness': cp['randomness'],
        'checkpoint_source_sha256': {name: sha(a.checkpoint_source/name) for name in ['checkpoint_schema.py','checkpoint_runtime.py']},
        'checkpoint_file_sha256': sha(a.checkpoint_run/'carry_checkpoint.npz'),
        'episodes': {f'precursor_{n}': audit_episode(a.bottle_root/f'precursor_{n}'/'episodes/A', a.bottle_root/f'precursor_{n}_capC_full') for n in range(1,5)},
        'fresh_isaac_replay': 'NOT_RUN_NVML_DRIVER_LIBRARY_MISMATCH',
        'source_mutations': 0,
        'limits': ['Historical carry success uses verified-grasp kinematic support', '30 settling ticks are excluded',
          'Whole-scene gate fails; only 19 of 56 rigid bodies tracked after settling',
          'Nonzero velocity is intentionally discarded by the archived adapter',
          'No cross-process end-to-end multi-stage physical chain proven',
          'Render replay is pose playback, not dynamic continuation', 'No new learner performance measured in this audit']}
    (a.output/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'stage_resume': result['overall_stage_resume'], 'serialization': serialization,
        'historical_carry': historical['status'], 'whole_scene': historical['whole_scene_resume_gate']['status'],
        'joint_velocity_preserved': probe['preserves_nonzero_joint_velocity'],
        'bottle_demonstrations': len(result['episodes']), 'source_mutations': 0}, indent=2))


if __name__ == '__main__':
    main()
