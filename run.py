#!/usr/bin/env python3
"""Reproducible PAR-CLIP conference-only experiment runner.

Four stages: scratch CLIP-style verifier, zero-shot CLIP baseline,
Zhang-style *adaptation* (NOT exact CVPR reproduction), proposed PAR-CLIP.
Every result is computed; nothing is hardcoded from the manuscript.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from methods import (unit, set_seed, scores, train_parclip, train_zhang_style, evaluate_parclip_state)

ImageFile.LOAD_TRUNCATED_IMAGES = True
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
AGE = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49',
       '50-59', '60-69', 'more than 70']
RACE = ['White', 'Black', 'Latino_Hispanic', 'East Asian',
        'Southeast Asian', 'Indian', 'Middle Eastern']
AGE_PROMPTS = [
    'A portrait photograph of a baby aged 0 to 2',
    'A portrait photograph of a child aged 3 to 9',
    'A portrait photograph of a person aged 10 to 19',
    'A portrait photograph of a person aged 20 to 29',
    'A portrait photograph of a person aged 30 to 39',
    'A portrait photograph of a person aged 40 to 49',
    'A portrait photograph of a person aged 50 to 59',
    'A portrait photograph of a person aged 60 to 69',
    'A portrait photograph of a person over 70 years old',
]
RACE_PROMPTS = [f'A portrait photograph of a {r.replace("_", " ")} person' for r in RACE]
# Neutral occupation probes; DPG is measured across age/race groups on these outputs.
OCCUPATIONS = [f'A photo of a {name}' for name in (
    'doctor', 'nurse', 'engineer', 'teacher', 'scientist', 'chef', 'lawyer', 'farmer',
    'cleaner', 'manager', 'secretary', 'pilot', 'construction worker', 'artist',
    'librarian', 'CEO')]


def config():
    with (ROOT / 'config.json').open() as f:
        return json.load(f)


def resolve_path(s):
    p = Path(os.path.expanduser(str(s)))
    if not p.is_absolute(): p = ROOT / p
    return p


def require(cfg, *fields):
    paths = [resolve_path(cfg[x]) for x in fields]
    missing = [f'{name}: {p}' for name, p in zip(fields, paths) if not p.exists()]
    if missing:
        raise FileNotFoundError('Configure existing dataset paths in config.json:\n' + '\n'.join(missing))
    return paths


def save_json(name, obj):
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / name).open('w') as f:
        json.dump(obj, f, indent=2, default=str)
    print('SAVED:', RESULTS / name)


def save_csv(name, rows):
    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / name, index=False)
    print('SAVED:', RESULTS / name)


def fairface_df(cfg, split, attribute, cap=None):
    img_root, labels_file = require(cfg, 'fairface_images', f'fairface_{split}_csv')
    df = pd.read_csv(labels_file)
    if not {'file', attribute}.issubset(df):
        raise ValueError(f'{labels_file} needs file and {attribute} columns')
    categories = AGE if attribute == 'age' else RACE
    mapping = {n: i for i, n in enumerate(categories)}
    df = df.assign(target=df[attribute].map(mapping))
    df = df.dropna(subset=['target']).copy()
    df['target'] = df['target'].astype(int)
    if cap is not None and len(df) > cap:
        df = df.sample(n=int(cap), random_state=int(cfg['seed']))
    df = df.reset_index(drop=True)
    df['image_path'] = df['file'].map(lambda n: str(img_root / str(n)))
    if not Path(df.loc[0, 'image_path']).is_file():
        raise FileNotFoundError('Images not found. fairface_images must be the folder containing train/ and val/: '\
                                + df.loc[0, 'image_path'])
    return df


class Images(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self): return len(self.frame)

    def __getitem__(self, ix):
        row = self.frame.iloc[ix]
        with Image.open(row['image_path']) as im:
            img = self.transform(im.convert('RGB'))
        return img, int(row['target'])


def clip_model(backbone):
    import open_clip
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable. Install CUDA-enabled PyTorch and check nvidia-smi.')
    print(f'Loading pretrained CLIP {backbone} (first download may take time)...')
    model, _, tfm = open_clip.create_model_and_transforms(backbone, pretrained='openai', device='cuda', force_quick_gelu=True)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    tok = open_clip.get_tokenizer(backbone)
    return model, tfm, tok


@torch.inference_mode()
def encode_images(model, transform, frame, cfg, title):
    bs = int(cfg['clip_batch_size'])
    loader = DataLoader(Images(frame, transform), batch_size=bs,
                        shuffle=False, num_workers=4, pin_memory=True)
    xs, ys = [], []
    for im, labels in tqdm(loader, desc=title):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            x = model.encode_image(im.cuda(non_blocking=True))
        xs.append(unit(x).cpu())
        ys.append(labels.cpu())
    return torch.cat(xs), torch.cat(ys)


@torch.inference_mode()
def encode_prompts(model, tokenizer, texts):
    out = []
    for i in range(0, len(texts), 64):
        toks = tokenizer(texts[i:i+64]).cuda()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            out.append(unit(model.encode_text(toks)).cpu())
    return torch.cat(out)


def cache_name(backbone, attr):
    return RESULTS / f'cache_QuickGELU_{backbone.replace("/", "-")}_{attr}.pt'


def cached_embeddings(cfg, attr, backbone):
    RESULTS.mkdir(exist_ok=True)
    cached = cache_name(backbone, attr)
    if cached.exists():
        print('Using cached embeddings:', cached)
        # Only load self-created tensor cache, not untrusted .pt files.
        return torch.load(cached, map_location='cpu', weights_only=True)
    train = fairface_df(cfg, 'train', attr, cfg['train_cap'])
    val = fairface_df(cfg, 'val', attr, cfg.get('val_cap'))
    print(f'{backbone} {attr}: {len(train)} train and {len(val)} val images')
    model, tfm, tok = clip_model(backbone)
    tr_x, tr_y = encode_images(model, tfm, train, cfg, 'Train embeddings')
    va_x, va_y = encode_images(model, tfm, val, cfg, 'Validation embeddings')
    cls = encode_prompts(model, tok, AGE_PROMPTS if attr == 'age' else RACE_PROMPTS)
    occ = encode_prompts(model, tok, OCCUPATIONS)
    pack = {'train_x': tr_x, 'train_y': tr_y, 'val_x': va_x, 'val_y': va_y,
            'class_prompts': cls, 'occupation_prompts': occ,
            'train_n': len(train), 'val_n': len(val), 'backbone': backbone,
            'attribute': attr, 'seed': int(cfg['seed'])}
    torch.save(pack, cached)
    del model
    torch.cuda.empty_cache()
    return pack


def stage2(cfg):
    rows = []
    for attr in ('age', 'race'):
        d = cached_embeddings(cfg, attr, cfg['embed_model'])
        m = scores(d['val_x'], d['class_prompts'], d['occupation_prompts'], d['val_y'])
        rows.append({'stage': '2_baseline', 'attribute': attr, 'backbone': d['backbone'], **m})
        print('Baseline', attr, m)
    save_csv('stage2_baseline.csv', rows)
    return rows


def stage3(cfg):
    rows = []
    for attr in ('age', 'race'):
        # Manuscript's base-paper reproduction uses CLIP ViT-B/16.
        d = cached_embeddings(cfg, attr, 'ViT-B-16')
        result, history, state = train_zhang_style(
            d['train_x'], d['train_y'], d['val_x'], d['val_y'],
            d['class_prompts'], d['occupation_prompts'],
            epochs=int(cfg['stage3_epochs']), seed=int(cfg['seed']))
        torch.save(state, RESULTS / f'stage3_zhang_STYLE_ONLY_{attr}.pt')
        save_csv(f'stage3_{attr}_history.csv', history)
        rows.append({'stage': '3_zhang_style_adaptation_NOT_official', 'attribute': attr,
                     'backbone': 'ViT-B-16', **result})
        print('Zhang STYLE adaptation (not an exact reproduction)', attr, result)
    save_csv('stage3_zhang_style_NOT_official.csv', rows)
    return rows


def stage4(cfg):
    """One orchestrated run; choose only on internal validation, then one holdout test."""
    rows = []
    for attr in ('age', 'race'):
        d = cached_embeddings(cfg, attr, cfg['embed_model'])
        y = d['train_y']
        fit_ix, sel_ix = train_test_split(np.arange(len(y)), test_size=.20,
                         stratify=y.numpy(), random_state=int(cfg['seed']))
        fit_ix = torch.as_tensor(fit_ix, dtype=torch.long)
        sel_ix = torch.as_tensor(sel_ix, dtype=torch.long)
        print(f'{attr}: fitting={len(fit_ix)} selection={len(sel_ix)} '
              f'final_holdout={len(d["val_y"])}', flush=True)
        candidates = cfg.get('projection_strength_candidates', [0.10, 0.30, 1.0])
        if not candidates or any(not 0 < float(x) <= 1 for x in candidates):
            raise ValueError('Invalid projection_strength_candidates')
        baseline_internal = scores(d['train_x'][sel_ix], d['class_prompts'],
                                   d['occupation_prompts'], y[sel_ix])
        evaluated = []
        selection_rows = []
        for beta in candidates:
            beta = float(beta)
            tag = str(beta).replace('.', 'p')
            cp = RESULTS / f'onepass_{attr}_beta{tag}_resume.pt'
            met, meta, hist, state = train_parclip(
                d['train_x'][fit_ix], y[fit_ix],
                d['train_x'][sel_ix], y[sel_ix],
                d['class_prompts'], d['occupation_prompts'],
                requested_k=int(cfg['requested_k'][attr]),
                fairness_weight=float(cfg['lambda_fair'][attr]),
                epochs=int(cfg['stage4_epochs']),
                batch_size=int(cfg['head_batch_size']),
                lr=float(cfg['head_learning_rate']), tau=float(cfg['tau']),
                seed=int(cfg['seed']),
                anchor=float(cfg['procrustes_identity_regularizer']),
                projection_strength=beta, checkpoint_path=cp)
            f, dp = met['macro_f1'], met['occupation_dpg']
            sfu = 2*f*(1-dp)/max(f+1-dp, 1e-12)
            guardrail = (met['accuracy'] >= baseline_internal['accuracy']-.02
                         and met['macro_f1'] >= baseline_internal['macro_f1']-.02)
            selection_rows.append({'attribute':attr, 'projection_strength':beta,
                 'effective_k': state['effective_k'], 'best_epoch':state['best_epoch'],
                 'accuracy_guardrail_pass':guardrail,
                 'internal_sfu':sfu, **met})
            evaluated.append((guardrail, sfu, met, meta, hist, state))
        save_csv(f'onepass_{attr}_internal_model_selection.csv', selection_rows)
        eligible = [v for v in evaluated if v[0]]
        pool = eligible if eligible else evaluated
        # If all fail guardrail: choose utility first, flag failure explicitly.
        selected = max(pool, key=(lambda v: (v[1], v[2]['accuracy']))
                       if eligible else (lambda v: (v[2]['macro_f1'], v[2]['accuracy'], v[1])))
        passed, _, internal, meta, hist, state = selected
        # Official FairFace validation dataset is only consulted *after*
        # model, epoch and projection strength have been selected.
        result = evaluate_parclip_state(state, d['val_x'], d['val_y'],
                                        d['class_prompts'], d['occupation_prompts'])
        torch.save(state, RESULTS / f'onepass_{attr}_selected_model.pt')
        save_csv(f'onepass_{attr}_training_history.csv', hist)
        save_json(f'onepass_{attr}_protocol.json', {
             'official_training_sample': len(y),
             'fit_n': len(fit_ix), 'internal_validation_n': len(sel_ix),
             'final_holdout_n': len(d['val_y']), 'seed':cfg['seed'],
             'requested_k':state['requested_k'], 'effective_k':state['effective_k'],
             'lambda_fair':state['lambda_fair'], 'selected_beta':state['projection_strength'],
             'selected_epoch':state['best_epoch'],
             'internal_accuracy_guardrail_pass':passed,
             'internal_baseline':baseline_internal,
             'internal_selected':internal,
             'test_evaluated_only_after_model_selection':True,
             'training_objective':'supervised image-to-demographic-prompt InfoNCE + neutral occupation parity',
             'note':'Partial projection (beta<1) is a methodological variant requiring disclosure; no paired COCO captions used.'})
        rows.append({'attribute':attr, 'stage':'4_PARCLIP_onepass',
             'projection_strength':state['projection_strength'],
             'requested_k':state['requested_k'], 'effective_k':state['effective_k'],
             'guardrail_pass_internal':passed, **result})
        print('FINAL OFFICIAL FAIRFACE HOLDOUT', attr, result, flush=True)
    save_csv('onepass_PARCLIP_holdout.csv', rows)
    paper_report()
    return rows


def paper_report():
    """Measured manuscript Table VI, comparisons, and Fig. 6 (no fabricated figures)."""
    bp, pp = RESULTS/'stage2_baseline.csv', RESULTS/'onepass_PARCLIP_holdout.csv'
    if not (bp.is_file() and pp.is_file()):
        print('Main paper table pending stage2 and stage4', flush=True)
        return
    b, p = pd.read_csv(bp), pd.read_csv(pp)
    rows = []
    for attr in ('age', 'race'):
        bb = b.loc[b.attribute == attr].iloc[0]
        pa = p.loc[p.attribute == attr].iloc[0]
        for method, vals in [('Baseline CLIP', bb), ('PAR-CLIP (internal-selected variant)', pa)]:
            f, d = float(vals.macro_f1), float(vals.occupation_dpg)
            rows.append({'attribute':attr, 'method':method,
              'accuracy':float(vals.accuracy), 'macro_f1':f, 'occupation_dpg':d,
              'S_FU':2*f*(1-d)/max(f+1-d, 1e-12),
              'F1_gain_percent':100*(float(pa.macro_f1)/float(bb.macro_f1)-1)
                 if method.startswith('PAR-CLIP') and float(bb.macro_f1)>0 else np.nan,
              'DPG_reduction_percent':100*(1-float(pa.occupation_dpg)/float(bb.occupation_dpg))
                 if method.startswith('PAR-CLIP') and float(bb.occupation_dpg)>0 else np.nan})
    save_csv('paper_table_VI_measured.csv', rows)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib missing: table saved, graph requires matplotlib', flush=True)
        return
    fig, ax = plt.subplots(1,3,figsize=(11,3.4))
    for axis, metric, title in zip(ax, ('accuracy','macro_f1','occupation_dpg'),
                                   ('Accuracy','Macro-F1','Occupation DPG')):
        for pos, method in enumerate(('Baseline CLIP','PAR-CLIP (internal-selected variant)')):
            vals = [next(float(v[metric]) for v in rows if v['attribute']==attr and v['method']==method)
                    for attr in ('age','race')]
            axis.bar(np.arange(2)+pos*.34-.17, vals, width=.34, label=method)
        axis.set_xticks([0,1], ['Age','Race'])
        axis.set_ylim(bottom=0)
        axis.set_title(title)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS/'paper_figure_6_measured.png',dpi=180)
    plt.close(fig)
    print('SAVED:', RESULTS/'paper_figure_6_measured.png', flush=True)


# ---------- Stage 1: custom, from-scratch CLIP-style PIPELINE verifier ----------

def caption_pairs(cfg, name):
    if name == 'flickr8k':
        images, txt = require(cfg, 'flickr8k_images', 'flickr8k_captions')
        pairs = []
        # Flickr8k commonly uses image,caption CSV or image#N<TAB>caption format.
        with txt.open(encoding='utf-8-sig') as f:
            first = f.readline().strip(); f.seek(0)
            if 'caption' in first.lower() and ('image' in first.lower()):
                reader = csv.DictReader(f)
                for row in reader:
                    im = row.get('image') or row.get('filename')
                    caption = row.get('caption') or row.get('comment')
                    if im and caption: pairs.append((str(images/im), str(caption)))
            else:
                for line in f:
                    cols = line.strip().split('\t',1)
                    if len(cols) == 2:
                        pairs.append((str(images/cols[0].split('#')[0]), cols[1]))
    else:
        images, ann = require(cfg, 'coco_images', 'coco_captions')
        with ann.open() as f: data = json.load(f)
        lookup = {x['id']: x['file_name'] for x in data['images']}
        pairs = [(str(images / lookup[a['image_id']]), a['caption'])
                 for a in data['annotations'] if a['image_id'] in lookup]
    # One caption per image: prevents same image appearing on both train and val.
    unique = {}
    for image_path, caption in pairs:
        if image_path not in unique and Path(image_path).is_file():
            unique[image_path] = caption
    result = list(unique.items())
    rng = np.random.default_rng(cfg['seed']); rng.shuffle(result)
    result = result[:int(cfg.get(f'stage1_{name}_max_pairs', cfg['stage1_max_pairs']))]
    if len(result) < 20:
        raise ValueError(f'Only {len(result)} valid {name} image-caption pairs; verify paths')
    return result


def tokenize(s): return re.findall(r"[a-z0-9']+", str(s).lower())[:30]


class PairedImages(Dataset):
    def __init__(self, pairs, vocab):
        from torchvision import transforms as T
        self.pairs = pairs
        self.vocab = vocab
        self.tfm = T.Compose([T.Resize((112,112)), T.ToTensor(),
                             T.Normalize((.5,.5,.5),(.5,.5,.5))])

    def __len__(self): return len(self.pairs)

    def __getitem__(self, i):
        path, txt = self.pairs[i]
        with Image.open(path) as f: img = self.tfm(f.convert('RGB'))
        seq = [self.vocab.get(w,1) for w in tokenize(txt)][:30]
        seq += [0] * (30-len(seq))
        return img, torch.tensor(seq, dtype=torch.long)


class TinyCLIP(nn.Module):
    """From-scratch ResNet18 + GRU CLIP verifier; actual params reported."""
    def __init__(self, vocab_size, dim=256):
        super().__init__()
        from torchvision.models import resnet18
        self.vision = resnet18(weights=None)
        self.vision.fc = nn.Linear(512, dim)
        self.embedding = nn.Embedding(vocab_size, 384, padding_idx=0)
        self.text_gru = nn.GRU(384, 512, batch_first=True)
        self.text_proj = nn.Linear(512, dim)

    def forward(self, im, seq):
        image = self.vision(im)
        length = (seq != 0).sum(1).clamp_min(1)
        packed = nn.utils.rnn.pack_padded_sequence(
            self.embedding(seq), length.cpu(), batch_first=True,
            enforce_sorted=False)
        _, h = self.text_gru(packed)
        return unit(image), unit(self.text_proj(h[-1]))


def stage1_one(cfg, name):
    """Image-disjoint 80/10/10 split, validation selection, test-once."""
    set_seed(int(cfg['seed']))
    pairs = caption_pairs(cfg, name)  # Unique images; 1 caption per image
    trainval, test = train_test_split(pairs, test_size=.10,
                                      random_state=int(cfg['seed']))
    train, val = train_test_split(trainval, test_size=1/9,
                                  random_state=int(cfg['seed']))
    if len({a for a, _ in train} & {a for a, _ in val+test}):
        raise RuntimeError('IMAGE LEAKAGE between partitions')
    vocab = {'<pad>': 0, '<unk>': 1}
    counts = Counter(w for _, cap in train for w in tokenize(cap))
    for word, cnt in counts.most_common(9998):
        if cnt >= 2:
            vocab[word] = len(vocab)
    model = TinyCLIP(len(vocab)).cuda()
    params = sum(x.numel() for x in model.parameters())
    print(f'{name}: train={len(train)}, val={len(val)}, test={len(test)}, '
          f'parameters={params:,} (not asserted to match original paper)', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    bs = int(cfg['stage1_batch_size'])
    loaders = [DataLoader(PairedImages(part, vocab), batch_size=batch,
                          shuffle=shuf, num_workers=2, pin_memory=True,
                          drop_last=drop)
               for part, batch, shuf, drop in
               ((train, bs, True, True), (val, 64, False, False),
                (test, 64, False, False))]
    train_loader, val_loader, test_loader = loaders
    history, best_loss, best_epoch, best_weights = [], float('inf'), 0, None
    stalled = 0
    latest = RESULTS / f'stage1_{name}_latest.pt'
    run_id = {'pairs': len(pairs), 'train': len(train), 'val':len(val),
              'test':len(test), 'vocab':len(vocab), 'seed':int(cfg['seed']),
              'batch':bs, 'param_count':params}
    epoch_start = 0
    if latest.exists():
        saved = torch.load(latest, map_location='cuda', weights_only=True)
        if saved['run_id'] != run_id:
            raise RuntimeError(f'Old stage1 checkpoint settings differ: {latest}')
        model.load_state_dict(saved['model'])
        opt.load_state_dict(saved['optimizer'])
        history = saved['history']
        best_loss, best_epoch = saved['best_loss'], saved['best_epoch']
        best_weights, stalled = saved['best_weights'], saved['stalled']
        epoch_start = saved['epoch']
        torch.set_rng_state(saved['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
        print(f'{name}: resuming after epoch {epoch_start}', flush=True)

    def validation_loss(loader):
        model.eval()
        total, batches = 0., 0
        with torch.inference_mode():
            for im, txt in loader:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    i, t = model(im.cuda(non_blocking=True), txt.cuda(non_blocking=True))
                    logits = i @ t.T / .07
                    y = torch.arange(len(i), device='cuda')
                    loss = (F.cross_entropy(logits, y)
                            + F.cross_entropy(logits.T, y))/2
                total += float(loss)
                batches += 1
        return total / max(1, batches)

    max_epochs = int(cfg['stage1_epochs'])
    patience = int(cfg.get('stage1_patience', 12))
    for ep in range(epoch_start, max_epochs):
        if stalled >= patience:
            break
        model.train()
        total, batches = 0., 0
        for im, txt in tqdm(train_loader, desc=f'{name} epoch {ep+1}/{max_epochs}'):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                i, t = model(im.cuda(non_blocking=True), txt.cuda(non_blocking=True))
                logits = i @ t.T / .07
                y = torch.arange(len(i), device='cuda')
                loss = (F.cross_entropy(logits, y)
                        + F.cross_entropy(logits.T, y))/2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss)
            batches += 1
        loss_val = validation_loss(val_loader)
        history.append({'epoch': ep+1, 'train_loss':total/max(1,batches),
                        'val_loss':loss_val})
        if loss_val < best_loss - 1e-4:
            best_loss, best_epoch, stalled = loss_val, ep+1, 0
            best_weights = {k:v.detach().cpu().clone()
                            for k,v in model.state_dict().items()}
        else:
            stalled += 1
        tmp = latest.with_suffix('.pt.tmp')
        torch.save({'run_id': run_id, 'epoch':ep+1,
                    'model':model.state_dict(), 'optimizer':opt.state_dict(),
                    'history':history, 'best_loss':best_loss,
                    'best_epoch':best_epoch, 'best_weights':best_weights,
                    'stalled':stalled, 'torch_rng':torch.get_rng_state(),
                    'cuda_rng':torch.cuda.get_rng_state_all()}, tmp)
        tmp.replace(latest)
        print(f'{name} epoch={ep+1} train={history[-1]["train_loss"]:.4f} '
              f'val={loss_val:.4f} best={best_epoch}', flush=True)
    if best_weights is None:
        raise RuntimeError('No trained best checkpoint')
    model.load_state_dict(best_weights)
    model.eval()
    img_emb, txt_emb = [], []
    with torch.inference_mode():
        for im, txt in test_loader:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                i, t = model(im.cuda(non_blocking=True), txt.cuda(non_blocking=True))
            img_emb.append(i.float().cpu())
            txt_emb.append(t.float().cpu())
    sim = unit(torch.cat(img_emb)) @ unit(torch.cat(txt_emb)).T
    n = len(sim)
    expected = torch.arange(n).unsqueeze(1)
    result = {'stage':'1_scratch_pipeline_new_architecture', 'dataset':name,
              'train_pairs':len(train), 'val_pairs':len(val), 'test_pairs':len(test),
              'best_epoch':best_epoch, 'max_epochs':max_epochs,
              'trainable_parameters':params,
              'image_to_text_R1':float((sim.topk(1, dim=1).indices==expected).any(1).float().mean()),
              'image_to_text_R5':float((sim.topk(min(5,n), dim=1).indices==expected).any(1).float().mean()),
              'text_to_image_R1':float((sim.T.topk(1, dim=1).indices==expected).any(1).float().mean()),
              'text_to_image_R5':float((sim.T.topk(min(5,n), dim=1).indices==expected).any(1).float().mean())}
    torch.save({'model':best_weights, 'vocab':vocab, 'configuration':run_id},
               RESULTS / f'stage1_{name}_scratch.pt')
    save_csv(f'stage1_{name}_history.csv', history)
    save_json(f'stage1_{name}_protocol.json', {**run_id,
        'best_epoch':best_epoch, 'max_epochs':max_epochs,
        'evaluation':'one caption per image; image-disjoint 80/10/10',
        'source':'New ResNet18+GRU architecture; NOT original manuscript architecture'})
    print('STAGE1 FINAL TEST:', result, flush=True)
    return result


def stage1(cfg):
    rows = [stage1_one(cfg, name) for name in ('flickr8k', 'coco')]
    save_csv('stage1_scratch_verification.csv', rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    keys = ('image_to_text_R1','image_to_text_R5',
            'text_to_image_R1','text_to_image_R5')
    xx = np.arange(len(rows))
    for i, key in enumerate(keys):
        ax.bar(xx + .2*(i-1.5), [row[key] for row in rows],
               width=.18, label=key)
    ax.set_xticks(xx, [row['dataset'] for row in rows])
    ax.set_ylabel('Recall on held-out images')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / 'stage1_retrieval_measured.png', dpi=180)
    plt.close(fig)
    print('SAVED:', RESULTS / 'stage1_retrieval_measured.png')
    return rows


def check(cfg):
    print('Project:', ROOT)
    print('CUDA:',torch.cuda.is_available(), 'GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not found')
    print('PyTorch:',torch.__version__)
    print('Paths:')
    for k,v in cfg.items():
        if k.endswith(('_images','_csv','_captions')):
            print(' ',k, ':', 'OK' if resolve_path(v).exists() else 'NOT FOUND',v)
    print('No files were modified.')


def main():
    parser=argparse.ArgumentParser(description='PAR-CLIP conference rerun (one folder)')
    parser.add_argument('stage',choices=['check','baseclip','baseline','zhang','parclip','all','final'])
    args=parser.parse_args()
    cfg=config(); set_seed(cfg['seed'])
    if args.stage=='check': return check(cfg)
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA detected: install GPU PyTorch first; see README.md')
    RESULTS.mkdir(exist_ok=True)
    save_json('config_snapshot.json',cfg)
    steps={'baseclip':stage1,'baseline':stage2,'zhang':stage3,'parclip':stage4}
    if args.stage=='final':
        stage2(cfg)
        stage4(cfg)
        return
    if args.stage=='all':
        for name,fn in steps.items():
            print('\n'+'='*20, name.upper(), '='*20)
            fn(cfg)
    else: steps[args.stage](cfg)
    print('\nFinished. All generated artifacts are in:', RESULTS)


if __name__=='__main__':
    main()
