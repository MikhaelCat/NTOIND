"""Strict external-holdout structured decoding for saved NLI probabilities.

Fits only on the training side of the fixed holdout. It never reads target labels.
Canonical labels: supported=0, contradicted=1, not_stated=2.
"""

import argparse
import difflib
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

LABELS = ('supported', 'contradicted', 'not_stated')
TOKEN_RE = re.compile(r'⟦e:[^⟧]+⟧|\d+(?:[.,]\d+)?|[а-яёa-z]{2,}', re.I)


def tokens(value):
    return TOKEN_RE.findall(str(value).lower())


def pair_features(left, right):
    a, b = tokens(left), tokens(right)
    sa, sb = set(a), set(b)
    ratio = difflib.SequenceMatcher(None, left.lower(), right.lower()).ratio()
    blocks = difflib.SequenceMatcher(None, a, b).get_matching_blocks()
    matched = sum(block.size for block in blocks)
    numbers_a = {x for x in sa if x[0].isdigit()}
    numbers_b = {x for x in sb if x[0].isdigit()}
    entities_a = {x for x in sa if x.startswith('⟦e:')}
    entities_b = {x for x in sb if x.startswith('⟦e:')}
    return [ratio, len(sa & sb) / max(1, len(sa | sb)), matched / max(1, len(a), len(b)),
            len(a), len(b), abs(len(a) - len(b)),
            len(numbers_a), len(numbers_b), len(numbers_a & numbers_b),
            len(entities_a), len(entities_b), len(entities_a & entities_b)]


def pair_type(first, second):
    return first * 3 + second if first <= second else second * 3 + first


def all_pairs(frame, labels=None):
    xs, ys, refs = [], [], []
    for _, group in frame.groupby('text', sort=False):
        rows = group.index.to_list()
        for i, j in itertools.combinations(rows, 2):
            xs.append(pair_features(frame.at[i, 'claim'], frame.at[j, 'claim']))
            refs.append((i, j))
            if labels is not None:
                ys.append(pair_type(labels[i], labels[j]))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int8), refs


def composition_key(values):
    return ''.join(map(str, sorted(values)))


def composition_priors(frame, labels):
    result = {}
    for _, group in frame.groupby('text', sort=False):
        key = len(group)
        result.setdefault(key, {})[composition_key(labels[group.index])] = result.setdefault(key, {}).get(composition_key(labels[group.index]), 0) + 1
    output = {}
    for size, counts in result.items():
        all_keys = [''.join(map(str, x)) for x in itertools.combinations_with_replacement(range(3), size)]
        total = sum(counts.values()) + 2 * len(all_keys)
        output[size] = {key: np.log((counts.get(key, 0) + 2) / total) for key in all_keys}
    return output


def find_invariants(claims):
    count = len(claims)
    candidates = []
    for i in range(count):
        for j in range(i + 1, count):
            for k in range(j + 1, count):
                sij = difflib.SequenceMatcher(None, claims[i], claims[j]).ratio()
                sik = difflib.SequenceMatcher(None, claims[i], claims[k]).ratio()
                sjk = difflib.SequenceMatcher(None, claims[j], claims[k]).ratio()
                if min(sij, sik, sjk) >= 0.70:
                    s_sums = [sij + sik, sij + sjk, sik + sjk]
                    med = [i, j, k][s_sums.index(max(s_sums))]
                    avg_s = (sij + sik + sjk) / 3
                    candidates.append((avg_s, i, j, k, med))
    candidates.sort(reverse=True)
    triplets = []
    used = set()
    for avg_s, i, j, k, med in candidates:
        if not (used & {i, j, k}):
            triplets.append((i, j, k, med))
            used.update([i, j, k])
    pairs_sim = []
    for i in range(count):
        for j in range(i + 1, count):
            if i not in used and j not in used:
                if difflib.SequenceMatcher(None, claims[i], claims[j]).ratio() >= 0.70:
                    pairs_sim.append((i, j))
    return triplets, pairs_sim


def decode_group(probabilities, pair_refs, priors, pair_weight, prior_weight,
                 triplets=None, pairs_sim=None, tb=1.5, sp=5.0, cp=1.0):
    count = len(probabilities)
    unary = np.log(np.clip(probabilities, 1e-8, 1))
    best_score, best = -np.inf, None
    triplets = triplets or []
    pairs_sim = pairs_sim or []
    for assignment in itertools.product(range(3), repeat=count):
        score = sum(unary[i, label] for i, label in enumerate(assignment))
        for left, right, values in pair_refs:
            score += pair_weight * values[pair_type(assignment[left], assignment[right])]
        score += prior_weight * priors.get(composition_key(assignment), -20.0)

        for i, j, k, med in triplets:
            t_labels = {assignment[i], assignment[j], assignment[k]}
            if t_labels == {0, 1, 2}:
                score += tb
                if assignment[med] == 0:
                    score += tb
            else:
                score -= 6.0

        for i, j in pairs_sim:
            if assignment[i] == 0 and assignment[j] == 0:
                score -= sp
            if assignment[i] == 1 and assignment[j] == 1:
                score -= cp

        if score > best_score:
            best_score, best = score, assignment
    return np.asarray(best, dtype=np.int8)


def decode(frame, probabilities, pair_model, pair_prior, priors, pair_weight=2.5, prior_weight=0.10,
           tb=1.5, sp=5.0, cp=1.0):
    output = np.empty(len(frame), dtype=np.int8)
    for _, group in frame.groupby('text', sort=False):
        original = group.index.to_list()
        local = group.reset_index(drop=True)
        claims = local['claim'].tolist()
        triplets, pairs_sim = find_invariants(claims)
        x, _, refs = all_pairs(local)
        if len(refs):
            pair_probability = pair_model.predict_proba(x)
            full = np.full((len(refs), 9), 1e-8, dtype=np.float64)
            full[:, pair_model.classes_] = pair_probability
            ratio = np.log(full) - pair_prior
            data = [(i, j, ratio[k]) for k, (i, j) in enumerate(refs)]
        else:
            data = []
        prediction = decode_group(probabilities[original], data, priors[len(group)],
                                  pair_weight, prior_weight,
                                  triplets=triplets, pairs_sim=pairs_sim, tb=tb, sp=sp, cp=cp)
        output[original] = prediction
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('.'))
    parser.add_argument('--probabilities', type=Path, required=True)
    parser.add_argument('--pair-weight', type=float, default=2.5)
    parser.add_argument('--prior-weight', type=float, default=0.10)
    parser.add_argument('--tb', type=float, default=1.5)
    parser.add_argument('--sp', type=float, default=5.0)
    parser.add_argument('--cp', type=float, default=1.0)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    archive = np.load(args.probabilities, allow_pickle=False)
    train = pd.read_csv(args.data / 'train.csv', dtype={'id': str})
    row_for_id = pd.Series(np.arange(len(train)), index=train.id)
    fit_rows = row_for_id.loc[archive['train_ids']].to_numpy()
    target_rows = row_for_id.loc[archive['target_ids']].to_numpy()
    fit = train.iloc[fit_rows].copy().reset_index(drop=True)
    target = train.iloc[target_rows].copy().reset_index(drop=True)
    y_fit = fit.label.map({name: index for index, name in enumerate(LABELS)}).to_numpy()
    y_target = target.label.map({name: index for index, name in enumerate(LABELS)}).to_numpy()
    pair_x, pair_y, _ = all_pairs(fit, y_fit)
    model = HistGradientBoostingClassifier(max_iter=110, learning_rate=.07, max_leaf_nodes=15,
                                           min_samples_leaf=25, random_state=19).fit(pair_x, pair_y)
    prior = np.full(9, np.log(1e-8), dtype=np.float64)
    counts = np.bincount(pair_y, minlength=9)
    prior[np.flatnonzero(counts)] = np.log(counts[np.flatnonzero(counts)] / counts.sum())
    priors = composition_priors(fit, y_fit)
    p_target = archive['target']
    raw = p_target.argmax(1)
    decoded = decode(target, p_target, model, prior, priors, args.pair_weight, args.prior_weight,
                     tb=args.tb, sp=args.sp, cp=args.cp)
    report = {
        'n_train_rows': int(len(fit)), 'n_train_posts': int(fit.text.nunique()),
        'n_target_rows': int(len(target)), 'n_target_posts': int(target.text.nunique()),
        'pair_weight': args.pair_weight, 'prior_weight': args.prior_weight,
        'raw_macro_f1': float(f1_score(y_target, raw, average='macro') * 100),
        'structured_macro_f1': float(f1_score(y_target, decoded, average='macro') * 100),
        'raw_errors': int((raw != y_target).sum()), 'structured_errors': int((decoded != y_target).sum()),
        'fixed': int(((raw != y_target) & (decoded == y_target)).sum()),
        'broken': int(((raw == y_target) & (decoded != y_target)).sum()),
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
