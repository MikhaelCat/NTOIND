#!/usr/bin/env python3
"""Unified entry point for NTO Case 2A (Fact-Checking).

Usage:
    python main.py                 # Train all models (if needed), then generate submission.csv
    python main.py --out file.csv  # Custom output path
    python main.py --train         # Force retrain all models on GPU
    python main.py --no-train      # Skip training, require pre-computed predictions
"""
import argparse
import hashlib
import json
import shutil
import subprocess
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

# -------------------------------------------------------------------------
# Конфигурации обучения — 5 моделей (точное воспроизведение оригинального ансамбля)
# -------------------------------------------------------------------------
# Первые три: peer_strong view (512 токенов)
# Последние две: full view — "Full-sequence baseline" из v14/v16
# -------------------------------------------------------------------------
TRAIN_CONFIGS = [
    {
        # mpeer: mdeberta с peer-контекстом (вес 0.35 в ансамбле)
        'key':     'mpeer',
        'model':   'mdeberta',
        'view':    'peer_strong',
        'max_len': 512,
        'lr':      1.5e-5,
        'epochs':  3,
        'batch':   16,
        'seed':    42,
        'out_dir': 'gpu_full_mdeberta_peerstrong512',
        'npz_key': 'target',   # читать из probabilities.npz['target']
    },
    {
        # xpeer: xlm с peer-контекстом (вес 0.20)
        'key':     'xpeer',
        'model':   'xlm',
        'view':    'peer_strong',
        'max_len': 512,
        'lr':      1.5e-5,
        'epochs':  3,
        'batch':   8,
        'seed':    42,
        'out_dir': 'gpu_full_xlm_peerstrong512',
        'npz_key': 'target',
    },
    {
        # rpeer: rubert с peer-контекстом (вес 0.10)
        'key':     'rpeer',
        'model':   'rubert',
        'view':    'peer_strong',
        'max_len': 512,
        'lr':      2.0e-5,
        'epochs':  3,
        'batch':   16,
        'seed':    42,
        'out_dir': 'gpu_full_rubert_peerstrong512',
        'npz_key': 'target',
    },
    {
        # mfull: mdeberta full-sequence baseline (вес 0.20)
        # Генерируется в gpu_full_mdeberta_full512/probabilities.npz
        # Итоговый файл копируется в mdeberta_test_probs.npy для обратной совместимости
        'key':     'mfull',
        'model':   'mdeberta',
        'view':    'full',
        'max_len': 512,
        'lr':      1.5e-5,
        'epochs':  3,
        'batch':   16,
        'seed':    42,
        'out_dir': 'gpu_full_mdeberta_full512',
        'npz_key': 'target',
    },
    {
        # xold: xlm full-sequence baseline (вес 0.15)
        # Генерируется в gpu_full_xlm_full512/probabilities.npz
        # Итоговый файл копируется в xlm_test_probs.npy для обратной совместимости
        'key':     'xold',
        'model':   'xlm',
        'view':    'full',
        'max_len': 512,
        'lr':      1.5e-5,
        'epochs':  3,
        'batch':   8,
        'seed':    42,
        'out_dir': 'gpu_full_xlm_full512',
        'npz_key': 'target',
    },
]

# Веса ансамбля (сумма = 1.0)
WEIGHTS = {
    'mpeer': 0.35,
    'xpeer': 0.20,
    'rpeer': 0.10,
    'mfull': 0.20,
    'xold':  0.15,
}

PAIR_WEIGHT  = 2.5
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


def predictions_ready() -> bool:
    """Return True if all required probabilities.npz files exist."""
    for cfg in TRAIN_CONFIGS:
        npz = pkg_dir / cfg['out_dir'] / 'probabilities.npz'
        if not npz.exists():
            return False
    return True


def run_training(data_dir: Path):
    """Train all models via train.py and produce probabilities.npz files."""
    train_script = pkg_dir / 'train.py'

    for cfg in TRAIN_CONFIGS:
        out_dir  = pkg_dir / cfg['out_dir']
        npz_path = out_dir / 'probabilities.npz'

        if npz_path.exists():
            print(f"[skip] {cfg['out_dir']}/probabilities.npz уже существует — пропускаем.")
            continue

        print(f"\n{'='*60}")
        print(f"[train] {cfg['model']} (view={cfg['view']}, lr={cfg['lr']}, epochs={cfg['epochs']}) ...")
        print(f"{'='*60}")

        cmd = [
            sys.executable, str(train_script),
            '--data',       str(data_dir),
            '--model',      cfg['model'],
            '--view',       cfg['view'],
            '--max-length', str(cfg['max_len']),
            '--lr',         str(cfg['lr']),
            '--epochs',     str(cfg['epochs']),
            '--batch-size', str(cfg['batch']),
            '--seed',       str(cfg['seed']),
            '--stage',      'full',
            '--out',        str(out_dir),
        ]
        print('CMD:', ' '.join(cmd))
        subprocess.run(cmd, check=True)
        print(f"[ok] {cfg['out_dir']}/probabilities.npz сохранён.")


def run_predict(data=None, test_file=None, out=Path('submission.csv'), report=None, force_train=False, no_train=False):
    root = data
    if root is None and test_file is not None:
        root = test_file.resolve().parent
    if root is None:
        root = pkg_dir if (pkg_dir / 'train.csv').exists() else pkg_dir.parent

    train_path  = root / 'train.csv'
    test_path   = test_file if test_file is not None else (root / 'test.csv')
    sample_path = root / 'sample_submission.csv'

    for p in (train_path, test_path, sample_path):
        if not p.exists():
            raise FileNotFoundError(f'Required input file not found: {p}')

    before_hashes = {p.name: sha256(p) for p in (train_path, test_path, sample_path)}

    # ------------------------------------------------------------------ #
    # Шаг 1: обучение (если нужно)                                        #
    # ------------------------------------------------------------------ #
    if force_train:
        print("[train] --train передан явно: запускаем полное переобучение.")
        run_training(root)
    elif not no_train and not predictions_ready():
        print("[train] Файлы предсказаний не найдены — запускаем обучение автоматически.")
        run_training(root)
    elif no_train and not predictions_ready():
        raise FileNotFoundError(
            "Файлы предсказаний не найдены, а флаг --no-train запрещает автообучение. "
            "Запустите без --no-train или сначала выполните python main.py --train"
        )

    # ------------------------------------------------------------------ #
    # Шаг 2: загрузка матриц вероятностей                                 #
    # ------------------------------------------------------------------ #
    train  = pd.read_csv(train_path,  dtype={'id': str}, encoding='utf-8')
    test   = pd.read_csv(test_path,   dtype={'id': str}, encoding='utf-8')
    sample = pd.read_csv(sample_path, dtype={'id': str}, encoding='utf-8')

    if list(sample.columns) != ['id', 'label']:
        raise ValueError('Unexpected sample_submission columns.')
    if not train.id.is_unique or not test.id.is_unique or not sample.id.is_unique:
        raise ValueError('Every input ID must be unique.')
    if not np.array_equal(test.id.to_numpy(), sample.id.to_numpy()):
        raise ValueError('test.csv IDs must equal sample_submission IDs in the same order.')

    arrays = {}
    for cfg in TRAIN_CONFIGS:
        npz_path = pkg_dir / cfg['out_dir'] / 'probabilities.npz'
        npz = np.load(npz_path, allow_pickle=False)
        arrays[cfg['key']] = npz[cfg['npz_key']]

    probabilities = sum(WEIGHTS[cfg['key']] * arrays[cfg['key']] for cfg in TRAIN_CONFIGS)

    if probabilities.shape != (len(test), 3) or not np.isfinite(probabilities).all():
        raise ValueError('Invalid ensemble probability matrix.')

    # ------------------------------------------------------------------ #
    # Шаг 3: структурный декод                                            #
    # ------------------------------------------------------------------ #
    y = train.label.map({'supported': 0, 'contradicted': 1, 'not_stated': 2}).to_numpy()
    feature, pair_labels, _ = decoder.all_pairs(train, y)
    pair_model = HistGradientBoostingClassifier(
        max_iter=110, learning_rate=0.07, max_leaf_nodes=15,
        min_samples_leaf=25, random_state=19,
    ).fit(feature, pair_labels)

    counts     = np.bincount(pair_labels, minlength=9)
    pair_prior = np.full(9, np.log(1e-8), dtype=np.float64)
    observed   = np.flatnonzero(counts)
    pair_prior[observed] = np.log(counts[observed] / counts.sum())

    prediction = decoder.decode(
        test.reset_index(drop=True), probabilities, pair_model, pair_prior,
        decoder.composition_priors(train, y),
        pair_weight=PAIR_WEIGHT, prior_weight=PRIOR_WEIGHT,
        tb=TB, sp=SP, cp=CP,
    )

    # ------------------------------------------------------------------ #
    # Шаг 4: сборка и валидация сабмита                                   #
    # ------------------------------------------------------------------ #
    output = pd.DataFrame({'id': sample.id, 'label': LABELS[prediction]})

    # Применение проверенных логико-семантических инвариантов
    try:
        import semantic_rules
        output = semantic_rules.apply_semantic_invariants(output)
    except Exception:
        pass

    if (len(output) != len(sample)
            or not output.id.is_unique
            or not np.array_equal(output.id.to_numpy(), sample.id.to_numpy())):
        raise RuntimeError('Submission IDs failed validation.')
    if not output.label.isin(LABELS).all():
        raise RuntimeError('Submission contains invalid labels.')

    after_hashes = {p.name: sha256(p) for p in (train_path, test_path, sample_path)}
    if after_hashes != before_hashes:
        raise RuntimeError('Input CSVs were modified during execution!')

    out = Path(out)
    output.to_csv(out, index=False, encoding='utf-8')
    print(f'\nSuccessfully built submission: {out} ({len(output)} rows)')
    print('Label distribution:\n', output['label'].value_counts().to_dict())

    # Дублируем в корень проекта для удобства
    try:
        parent_out = pkg_dir.parent / out.name
        if out.resolve() != parent_out.resolve() and (pkg_dir.parent / 'test.csv').exists():
            shutil.copy2(out, parent_out)
    except Exception:
        pass

    if report is not None:
        rep = {
            'candidate':    str(out.name),
            'rows':         int(len(output)),
            'weights':      WEIGHTS,
            'label_counts': output.label.value_counts().sort_index().to_dict(),
            'sha256':       sha256(out),
            'input_sha256': before_hashes,
        }
        Path(report).write_text(json.dumps(rep, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Generate final NTO Case 2A submission.')
    parser.add_argument('--data', '--data_dir', dest='data', type=Path, default=None,
                        help='Directory containing train.csv, test.csv, sample_submission.csv')
    parser.add_argument('--test', '--test_file', dest='test_file', type=Path, default=None,
                        help='Path to test.csv')
    parser.add_argument('--out', '--out_file', '--output', dest='out', type=Path, default=Path('submission.csv'),
                        help='Output submission CSV path')
    parser.add_argument('--report', type=Path, default=None, help='Output report JSON path')

    train_group = parser.add_mutually_exclusive_group()
    train_group.add_argument('--train', dest='force_train', action='store_true',
                             help='Force retrain all models even if predictions already exist')
    train_group.add_argument('--no-train', dest='no_train', action='store_true',
                             help='Do not train; fail if predictions are missing')

    args = parser.parse_args()

    run_predict(
        data=args.data,
        test_file=args.test_file,
        out=args.out,
        report=args.report,
        force_train=args.force_train,
        no_train=args.no_train,
    )


if __name__ == '__main__':
    main()
