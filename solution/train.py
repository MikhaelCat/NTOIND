import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import hashlib
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

LABELS = ('supported', 'contradicted', 'not_stated')
MODELS = {
    'mdeberta': 'mdeberta-v3-base-xnli',
    'xlm': 'xlm-roberta-large-xnli',
    'rubert': 'rubert-base-cased-nli-threeway',
}
HF_MODELS = {
    'mdeberta': 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7',
    'xlm': 'joeddav/xlm-roberta-large-xnli',
    'rubert': 'cointegrated/rubert-base-cased-nli-threeway',
}
MODEL_LABELS = ('entailment', 'contradiction', 'neutral')


def input_hashes(data_dir):
    names = ('train.csv', 'test.csv', 'sample_submission.csv')
    result = {}
    for name in names:
        p = data_dir / name
        if not p.exists():
            continue
        digest = hashlib.sha256()
        with p.open('rb') as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(chunk)
        result[name] = digest.hexdigest()
    return result


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fixed_folds(frame, count):
    folds = np.full(len(frame), -1, dtype=np.int8)
    for fold, (_, validation) in enumerate(GroupKFold(count).split(frame, groups=frame.text)):
        folds[validation] = fold
    return folds


def batches(rows, lengths, size, rng=None):
    rows = np.asarray(rows)
    if rng is None:
        return [rows[i:i + size] for i in range(0, len(rows), size)]
    rows = rng.permutation(rows)
    result = []
    for i in range(0, len(rows), 24 * size):
        pool = rows[i:i + 24 * size]
        pool = pool[np.argsort(lengths[pool], kind='stable')]
        result.extend(pool[j:j + size] for j in range(0, len(pool), size))
    rng.shuffle(result)
    return result


def evidence_text(text, claim, neighbors=1, maximum_sentences=4):
    token_re = re.compile(r'⟦e:[^⟧]+⟧|\d+(?:[.,]\d+)?|[а-яёa-z]{3,}', re.I)
    sent_re = re.compile(r'(?<=[.!?…])\s+|\n+')
    stopwords = {'это', 'этот', 'этим', 'этом', 'этого', 'также', 'который', 'которая',
                 'которые', 'которых', 'были', 'была', 'было', 'будет', 'является',
                 'являются', 'среди', 'после', 'перед', 'через', 'около', 'между', 'для',
                 'при', 'как', 'или', 'что', 'его', 'её', 'их', 'она', 'они', 'оно',
                 'когда', 'если', 'тогда', 'только', 'уже', 'ещё', 'более', 'менее'}
    sentences = [p.strip() for p in sent_re.split(str(text)) if p.strip()] or [str(text)]
    claim_tokens = [t.lower() for t in token_re.findall(str(claim)) if t.lower() not in stopwords]
    if not claim_tokens or len(sentences) <= maximum_sentences:
        return ' '.join(sentences)
    def score(sent_tokens):
        s_set = set(sent_tokens)
        return sum(t in s_set for t in claim_tokens) + 2.5 * sum(t.startswith('⟦e:') and t in s_set for t in claim_tokens)
    tokenized = [[t.lower() for t in token_re.findall(s) if t.lower() not in stopwords] for s in sentences]
    scores = [score(toks) for toks in tokenized]
    ranked = sorted(range(len(sentences)), key=lambda i: (-scores[i], i))[:maximum_sentences]
    selected = set()
    for idx in ranked:
        selected.update(range(max(0, idx - neighbors), min(len(sentences), idx + neighbors + 1)))
    ordered = sorted(selected)[:maximum_sentences + 2 * neighbors]
    return ' '.join(sentences[i] for i in ordered)


def encode(tokenizer, frame, view, max_length):
    texts = frame.text.tolist()
    claims = frame.claim.tolist()
    if view == 'evidence':
        texts = [evidence_text(text, claim) for text, claim in zip(texts, claims)]
    elif view in {'peer', 'peer_strong'}:
        import difflib
        peer_by_row = {}
        for _, group in frame.groupby('text', sort=False):
            indices = group.index.to_list()
            for index in indices:
                candidates = [candidate for candidate in indices if candidate != index]
                if candidates:
                    peer = max(candidates, key=lambda candidate: difflib.SequenceMatcher(
                        None, str(frame.at[index, 'claim']).lower(), str(frame.at[candidate, 'claim']).lower()
                    ).ratio())
                    similarity = difflib.SequenceMatcher(
                        None, str(frame.at[index, 'claim']).lower(), str(frame.at[peer, 'claim']).lower()
                    ).ratio()
                    peer_by_row[index] = str(frame.at[peer, 'claim']) if view == 'peer' or similarity >= .85 else ''
                else:
                    peer_by_row[index] = ''
        claims = [f'[Целевое утверждение] {claim} [Похожее утверждение] {peer_by_row[index]}'
                  for index, claim in zip(frame.index, claims)]
    elif view == 'multipeer':
        import difflib
        multipeer_by_row = {}
        for _, group in frame.groupby('text', sort=False):
            indices = group.index.to_list()
            for idx in indices:
                c = str(frame.at[idx, 'claim'])
                candidates = []
                for other in indices:
                    if other != idx:
                        sim = difflib.SequenceMatcher(None, c.lower(), str(frame.at[other, 'claim']).lower()).ratio()
                        if sim >= 0.65:
                            candidates.append((sim, str(frame.at[other, 'claim'])))
                candidates.sort(key=lambda x: x[0], reverse=True)
                if len(candidates) >= 2:
                    multipeer_by_row[idx] = f' [Вариант 1] {candidates[0][1]} [Вариант 2] {candidates[1][1]}'
                elif len(candidates) == 1:
                    multipeer_by_row[idx] = f' [Вариант 1] {candidates[0][1]}'
                else:
                    multipeer_by_row[idx] = ''
        claims = [f'[Целевое утверждение] {claim}{multipeer_by_row[idx]}'
                  for idx, claim in zip(frame.index, claims)]

    result = tokenizer(texts, claims, truncation='only_first', max_length=max_length,
                       return_token_type_ids=False, return_attention_mask=False)['input_ids']
    return result


def gpu_batch(encoded, indices, pad_id):
    seqs = [encoded[i] for i in indices]
    length = (max(map(len, seqs)) + 7) // 8 * 8
    ids = np.full((len(seqs), length), pad_id, dtype=np.int64)
    mask = np.zeros_like(ids)
    for i, sequence in enumerate(seqs):
        ids[i, :len(sequence)] = sequence
        mask[i, :len(sequence)] = 1
    return {'input_ids': torch.from_numpy(ids).cuda(), 'attention_mask': torch.from_numpy(mask).cuda()}


@torch.inference_mode()
def predict(model, encoded, rows, size, pad_id, canonical):
    model.eval()
    result = []
    for indices in batches(rows, None, size):
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(**gpu_batch(encoded, indices, pad_id)).logits
        result.append(logits.float().softmax(dim=-1)[:, canonical].cpu().numpy())
    return np.concatenate(result)


def log_score(y, probabilities):
    return 100 * f1_score(y, probabilities.argmax(axis=1), average='macro', labels=[0, 1, 2])


def fit_model(args, model_path, encoded, pad_id, labels, train_rows, stage, fold, local_only=True):
    seed = args.seed + fold
    seed_all(seed)
    kwargs = {'attn_implementation': 'sdpa'} if args.model == 'xlm' else {}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=local_only, dtype=torch.float32, **kwargs)
    lookup = {str(name).lower(): int(i) for i, name in model.config.id2label.items()}
    canonical = [lookup[label] for label in MODEL_LABELS]
    y_cuda = torch.as_tensor(np.asarray(canonical)[labels], device='cuda')
    model.cuda()
    del kwargs
    no_decay = [p for name, p in model.named_parameters() if name.endswith('bias') or p.ndim == 1]
    decay = [p for name, p in model.named_parameters() if not (name.endswith('bias') or p.ndim == 1)]
    optimizer = torch.optim.AdamW([{'params': decay, 'weight_decay': .01},
                                   {'params': no_decay, 'weight_decay': 0.}],
                                  lr=args.lr, eps=1e-6, fused=True)
    steps = ((len(train_rows) + args.batch_size - 1) // args.batch_size) * args.epochs
    schedule = get_cosine_schedule_with_warmup(optimizer, round(steps * .08), steps)
    scaler = torch.amp.GradScaler('cuda', init_scale=4096)
    lengths = np.fromiter(map(len, encoded), dtype=np.int32)
    rng = np.random.default_rng(seed)
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epochs):
        start = time.monotonic()
        model.train()
        total = torch.zeros((), device='cuda')
        skipped = 0
        parts = batches(train_rows, lengths, args.batch_size, rng)
        for step, indices in enumerate(parts, 1):
            optimizer.zero_grad(set_to_none=True)
            target = y_cuda[torch.as_tensor(indices, device='cuda')]
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(**gpu_batch(encoded, indices, pad_id)).logits
                loss = torch.nn.functional.cross_entropy(logits, target, label_smoothing=.03)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                schedule.step()
            else:
                skipped += 1
            total += loss.detach()
            if step == 1 or step % 100 == 0:
                print(f'{stage} fold={fold} epoch={epoch + 1} step={step}/{len(parts)} '
                      f'loss={loss.item():.4f} device={logits.device}', flush=True)
        average = total.item() / len(parts)
        if not np.isfinite(average) or skipped > len(parts) / 4:
            raise RuntimeError(f'Unstable FP16: loss={average} skipped={skipped}/{len(parts)}')
        print(f'{stage} fold={fold} epoch={epoch + 1}/{args.epochs} loss={average:.4f} '
              f'skipped={skipped} sec={time.monotonic()-start:.1f} '
              f'peak_reserved_gib={torch.cuda.max_memory_reserved()/2**30:.2f}', flush=True)
    return model, canonical


def save_atomic(path, **arrays):
    temp = path.with_suffix('.tmp.npz')
    np.savez_compressed(temp, **arrays)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description='Train multi-peer transformer models for NTO Case 2A.')
    parser.add_argument('--data', type=Path, default=None, help='Directory containing train.csv, test.csv')
    parser.add_argument('--models', type=Path, help='Path to directory with models')
    parser.add_argument('--out', type=Path, required=True, help='Output directory for predictions and checkpoints')
    parser.add_argument('--model', choices=MODELS, default='mdeberta')
    parser.add_argument('--view', choices=['full', 'evidence', 'peer', 'peer_strong', 'multipeer'], default='multipeer')
    parser.add_argument('--stage', choices=['strict', 'full'], default='full')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1.5e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--only-fold', type=int, default=-1)
    args = parser.parse_args()

    data_dir = args.data
    if data_dir is None:
        for cand in [Path('.'), Path('..'), Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent]:
            if (cand / 'train.csv').exists():
                data_dir = cand
                break
    if data_dir is None:
        data_dir = Path('.')

    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required; no CPU training fallback.')
    source_hashes = input_hashes(data_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    print(f'Device: {torch.cuda.get_device_name(0)}; offline FP16; {args}', flush=True)
    
    train = pd.read_csv(data_dir / 'train.csv', dtype={'id': str}, encoding='utf-8')
    test = pd.read_csv(data_dir / 'test.csv', dtype={'id': str}, encoding='utf-8')
    labels = np.array([LABELS.index(x) for x in train.label], dtype=np.int64)
    outer = fixed_folds(train, 5)
    
    models_dir = args.models
    if models_dir is None:
        for candidate in [data_dir / 'models', Path('models'), Path('../models'), Path(__file__).resolve().parent / 'models', Path(__file__).resolve().parent.parent / 'models']:
            if (candidate / MODELS[args.model]).exists():
                models_dir = candidate
                break
    if models_dir is None:
        models_dir = data_dir / 'models'
    local_path = models_dir / MODELS[args.model]
    if local_path.exists():
        model_path = str(local_path)
        local_only = True
        print(f'Используем локальные веса: {model_path}', flush=True)
    else:
        model_path = HF_MODELS[args.model]
        local_only = False
        print(f'Локальные веса не найдены в {local_path}. Автоматическая загрузка из Hugging Face Hub: {model_path}...', flush=True)
        
    print(f'Loading model from {model_path} (local_only={local_only})...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_only)
    print('Encoding train texts...', flush=True)
    train_encoded = encode(tokenizer, train, args.view, args.max_length)
    print('Encoding test texts...', flush=True)
    test_encoded = encode(tokenizer, test, args.view, args.max_length)
    pad_id = tokenizer.pad_token_id or 0
    test_rows = np.arange(len(test))

    splits = [(np.flatnonzero(outer != f), np.flatnonzero(outer == f), None) for f in range(5)]
    oof = np.full((len(train), 3), np.nan, dtype=np.float32)
    target_predictions = []

    for fold, (train_rows, val_rows, _) in enumerate(splits):
        if args.only_fold >= 0 and fold != args.only_fold:
            continue
        print(f'\n=== Training Fold {fold} ===', flush=True)
        model, order = fit_model(args, model_path, train_encoded, pad_id, labels, train_rows, args.stage, fold, local_only=local_only)
        size = args.batch_size * 2
        oof[val_rows] = predict(model, train_encoded, val_rows, size, pad_id, order)
        score = log_score(labels[val_rows], oof[val_rows])
        print(f'Fold {fold}: Raw Macro-F1 = {score:.4f}', flush=True)
        pred = predict(model, test_encoded, test_rows, size, pad_id, order)
        target_predictions.append(pred)
        save_atomic(args.out / f'fold_{fold}.npz', validation_ids=train.id.iloc[val_rows].to_numpy(dtype=str),
                    validation_probs=oof[val_rows], target_probs=pred)
        del model
        torch.cuda.empty_cache()

    if args.only_fold < 0:
        save_atomic(args.out / 'probabilities.npz', train_ids=train.id.to_numpy(dtype=str),
                    oof=oof, target_ids=test.id.to_numpy(dtype=str), target=np.mean(target_predictions, axis=0),
                    folds=outer, labels=np.array(LABELS))
        print(f'Overall OOF Raw Macro-F1 = {log_score(labels, oof):.4f}', flush=True)


if __name__ == '__main__':
    main()
