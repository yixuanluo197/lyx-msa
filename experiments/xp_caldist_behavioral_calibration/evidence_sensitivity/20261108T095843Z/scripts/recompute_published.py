"""Recompute published Validation statistics in a fresh CPU-only output directory.

Publication adapter only: frozen inference, analysis and source files are unchanged.
Full external source/weight auditing still requires the original local environment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CSVS = ['per_utterance_sensitivity.csv', 'modality_sensitivity_summary.csv',
        'sensitivity_summary.csv', 'classwise_sensitivity.csv',
        'correctness_group_contrasts.csv', 'previous_pilot_sensitivity_comparison.csv',
        'visual_missingness_analysis.csv', 'high_confidence_wrong_sensitive_cases.csv',
        'monotonicity_analysis.csv']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for part in iter(lambda: handle.read(4*1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Fresh directory; must not exist')
    args = parser.parse_args()
    target = args.output.expanduser().resolve()
    if target.exists() or target == ROOT or ROOT in target.parents:
        parser.error('--output must be a new directory outside the frozen experiment')
    previous = ROOT.parents[1] / '20261107T172302Z'
    if not (previous/'results/utterance_behavior_metrics.csv').is_file():
        parser.error('The preceding published pilot must be present beside evidence_sensitivity/')
    # Prevent a new output from being placed inside that frozen preceding pilot.
    if previous == target or previous in target.parents:
        parser.error('--output must be outside the preceding frozen pilot')
    shutil.copytree(ROOT, target)
    for relative in ['verification.json', 'provenance/artifact_manifest.json',
                     'provenance/source_hashes.json', 'publication_manifest.json',
                     'publication_verification.json']:
        path = target/relative
        if path.exists():
            reference = target/'provenance/published_reference'/relative
            reference.parent.mkdir(parents=True, exist_ok=True)
            path.rename(reference)
    config_path = target/'configs/evidence_sensitivity_config.json'
    config = json.loads(config_path.read_text())
    config['previous_pilot'] = str(previous)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    command = [sys.executable, str(target/'scripts/analyze_evidence.py'), '--root', str(target)]
    subprocess.run(command, env=env, check=True)
    csv_checks = {name: sha(target/'results'/name) == sha(ROOT/'results'/name) for name in CSVS}
    import numpy as np
    weight_checks = {}
    for dataset in config['anchors']:
        relative = f'results/{dataset}_paired_bootstrap_weights.npz'
        with np.load(ROOT/relative, allow_pickle=False) as original, np.load(target/relative, allow_pickle=False) as rebuilt:
            weight_checks[dataset] = (set(original.files) == set(rebuilt.files) and
                                     all(np.array_equal(original[key], rebuilt[key]) for key in original.files))
    result = dict(status='passed' if all(csv_checks.values()) and all(weight_checks.values()) else 'failed',
                  split='validation', model_forward=False, calibration_fitting=False,
                  csv_byte_matches=csv_checks, bootstrap_array_matches=weight_checks,
                  note='Offline prediction-based verification; not a new audit of inaccessible data/weights. '
                       'Reference provenance is retained under provenance/published_reference/. '
                       'Manual stable-missing report supplement is retained in the published original reports/logs.')
    (target/'offline_recomputation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
