#!/usr/bin/env python3
"""Unified entry point for NTO Case 2A (Fact-Checking).

Usage:
    python main.py                 # Generate submission.csv (< 30s)
    python main.py --out file.csv  # Custom output path
    python main.py --train         # Retrain models on GPU
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

pkg_dir = Path(__file__).resolve().parent
if str(pkg_dir) not in sys.path:
    sys.path.insert(0, str(pkg_dir))

import strict_structured_decode as decoder

LABELS = np.array(['supported', 'contradicted', 'not_stated'])
WEIGHTS = {'mpeer': 0.35, 'xpeer': 0.20, 'rpeer': 0.10, 'mfull': 0.20, 'xold': 0.15}
PAIR_WEIGHT = 2.5
PRIOR_WEIGHT = 0.10
TB = 1.5
SP = 5.0
CP = 1.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def run_predict(data=None, test_file=None, out=Path('submission.csv'), report=None):
    root = data
    if root is None and test_file is not None:
        root = test_file.resolve().parent
    if root is None:
        root = pkg_dir if (pkg_dir / 'train.csv').exists() else pkg_dir.parent

    train_path = root / 'train.csv'
    test_path = test_file if test_file is not None else (root / 'test.csv')
    sample_path = root / 'sample_submission.csv'

    for p in (train_path, test_path, sample_path):
        if not p.exists():
            raise FileNotFoundError(f'Required input file not found: {p}')

    before_hashes = {p.name: sha256(p) for p in (train_path, test_path, sample_path)}

    train = pd.read_csv(train_path, dtype={'id': str}, encoding='utf-8')
    test = pd.read_csv(test_path, dtype={'id': str}, encoding='utf-8')
    sample = pd.read_csv(sample_path, dtype={'id': str}, encoding='utf-8')

    if list(sample.columns) != ['id', 'label']:
        raise ValueError('Unexpected sample_submission columns.')
    if not train.id.is_unique or not test.id.is_unique or not sample.id.is_unique:
        raise ValueError('Every input ID must be unique.')
    if not np.array_equal(test.id.to_numpy(), sample.id.to_numpy()):
        raise ValueError('test.csv IDs must equal sample_submission IDs in the same order.')

    arrays = {
        'mpeer': np.load(pkg_dir / 'gpu_full_mdeberta_peerstrong512' / 'probabilities.npz', allow_pickle=False),
        'xpeer': np.load(pkg_dir / 'gpu_full_xlm_peerstrong512' / 'probabilities.npz', allow_pickle=False),
        'rpeer': np.load(pkg_dir / 'gpu_full_rubert_peerstrong512' / 'probabilities.npz', allow_pickle=False),
        'mfull': np.load(pkg_dir / 'mdeberta_test_probs.npy'),
        'xold': np.load(pkg_dir / 'xlm_test_probs.npy'),
    }

    probabilities = (
        WEIGHTS['mpeer'] * arrays['mpeer']['target']
        + WEIGHTS['xpeer'] * arrays['xpeer']['target']
        + WEIGHTS['rpeer'] * arrays['rpeer']['target']
        + WEIGHTS['mfull'] * arrays['mfull']
        + WEIGHTS['xold'] * arrays['xold']
    )

    if probabilities.shape != (len(test), 3) or not np.isfinite(probabilities).all():
        raise ValueError('Invalid ensemble probability matrix.')

    # стурктурный декод
    y = train.label.map({'supported': 0, 'contradicted': 1, 'not_stated': 2}).to_numpy()
    feature, pair_labels, _ = decoder.all_pairs(train, y)
    pair_model = HistGradientBoostingClassifier(
        max_iter=110, learning_rate=0.07, max_leaf_nodes=15,
        min_samples_leaf=25, random_state=19,
    ).fit(feature, pair_labels)

    counts = np.bincount(pair_labels, minlength=9)
    pair_prior = np.full(9, np.log(1e-8), dtype=np.float64)
    observed = np.flatnonzero(counts)
    pair_prior[observed] = np.log(counts[observed] / counts.sum())

    prediction = decoder.decode(
        test.reset_index(drop=True), probabilities, pair_model, pair_prior,
        decoder.composition_priors(train, y),
        pair_weight=PAIR_WEIGHT, prior_weight=PRIOR_WEIGHT,
        tb=TB, sp=SP, cp=CP,
    )

    output = pd.DataFrame({'id': sample.id, 'label': LABELS[prediction]})

    # валидация
    if len(output) != len(sample) or not output.id.is_unique or not np.array_equal(output.id.to_numpy(), sample.id.to_numpy()):
        raise RuntimeError('Submission IDs failed validation.')
    if not output.label.isin(LABELS).all():
        raise RuntimeError('Submission contains invalid labels.')

    after_hashes = {p.name: sha256(p) for p in (train_path, test_path, sample_path)}
    if after_hashes != before_hashes:
        raise RuntimeError('Input CSVs were modified during execution!')

    out = Path(out)
    output.to_csv(out, index=False, encoding='utf-8')
    print(f'Successfully built submission: {out} ({len(output)} rows)')
    print('Label distribution:\n', output['label'].value_counts().to_dict())

    try:
        parent_out = pkg_dir.parent / out.name
        if out.resolve() != parent_out.resolve() and (pkg_dir.parent / 'test.csv').exists():
            shutil.copy2(out, parent_out)
    except Exception:
        pass

    if report is not None:
        rep = {
            'candidate': str(out.name),
            'rows': int(len(output)),
            'weights': WEIGHTS,
            'label_counts': output.label.value_counts().sort_index().to_dict(),
            'sha256': sha256(out),
            'input_sha256': before_hashes,
        }
        Path(report).write_text(json.dumps(rep, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    if '--train' in sys.argv:
        import train
        sys.argv.remove('--train')
        train.main()
        return

    parser = argparse.ArgumentParser(description='Generate final NTO Case 2A submission.')
    parser.add_argument('--data', '--data_dir', dest='data', type=Path, default=None,
                        help='Directory containing train.csv, test.csv, sample_submission.csv')
    parser.add_argument('--test', '--test_file', dest='test_file', type=Path, default=None,
                        help='Path to test.csv')
    parser.add_argument('--out', '--out_file', '--output', dest='out', type=Path, default=Path('submission.csv'),
                        help='Output submission CSV path')
    parser.add_argument('--report', type=Path, default=None, help='Output report JSON path')
    args = parser.parse_args()

    run_predict(data=args.data, test_file=args.test_file, out=args.out, report=args.report)


if __name__ == '__main__':
    main()
