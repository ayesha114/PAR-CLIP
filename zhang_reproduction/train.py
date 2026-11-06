#!/usr/bin/env python3
"""Zhang et al. (CVPR'25) matched-budget embedding-level reproduction.

Applies Zhang's two-MLP residual architecture, momentum copies, 65,536-item
feature queues, counterfactual text inputs and Eq-style losses to the SAME
cached normalized CLIP embeddings used for PAR-CLIP; NOT an exact reproduction
of the full image-encoder training pipeline or published ABLE/MaxSkew.

Training budget: ViT-B/32 OpenAI QuickGELU, FairFace 16k/4k internal and
10,954 official FairFace validation, 30 epochs, batch 256, LR=1e-4, seed=42.
Zhang-specific alpha=.3, momentum=.995, queue_size=65536, mix_ratio=.4.

All artifacts reside in ~/PAR_CLIP_Conference/zhang_reproduction.
No existing PAR-CLIP scripts, weights or metrics are modified.
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run  # noqa: E402
from methods import scores, set_seed, unit  # noqa: E402


def json_save(path, obj):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=str) + '\n')
    tmp.replace(path)


def csv_save(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def original_age_label(age):
    return ('young' if age in ('0-2', '3-9', '10-19', '20-29')
            else 'middle-aged' if age in ('30-39', '40-49') else 'old')


def descriptions(frame, attr, seed):
    """Zhang's contextual FairFace prompts and alternate counterfactuals."""
    rng = random.Random(seed)
    all_races = ['White', 'Southeast Asian', 'Middle Eastern', 'Black',
                 'Indian', 'Latino_Hispanic', 'East Asian']
    txt1, txt2 = [], []
    for row in frame.itertuples(index=False):
        age, race, gender = row.age, row.race, row.gender.lower()
        if attr == 'age':
            age_group = original_age_label(age)
            other = rng.choice([a for a in ('young', 'middle-aged', 'old')
                                if a != age_group])
            txt1.append(f'This is the photo of a {age_group} {race} {gender}.')
            txt2.append(f'This is the photo of a {other} {race} {gender}.')
        else:
            other = rng.choice([r for r in all_races if r != race])
            txt1.append(f'This is the photo of a {age} {race} {gender}.')
            txt2.append(f'This is the photo of a {age} {other} {gender}.')
    return txt1, txt2


@torch.inference_mode()
def load_or_make_text_features(attr, cfg, frame):
    """Cache text1/text2 corresponding to training rows; does not touch images."""
    cache = HERE / f'{attr}_counterfactual_text_cache.pt'
    texts1, texts2 = descriptions(frame, attr, int(cfg['seed']))
    # A signature tied to the ordered labels and exact prompts avoids stale reuse.
    import hashlib
    signature = hashlib.sha256('\n'.join(
        a+'\t'+b for a, b in zip(texts1, texts2)).encode()).hexdigest()
    if cache.exists():
        d = torch.load(cache, map_location='cpu', weights_only=True)
        if d['signature'] == signature:
            return d['t1'].float(), d['t2'].float()
        raise RuntimeError(f'Cached counterfactual captions mismatch: {cache}. '
                           'Do not mix dataset selections; inspect the cache.')
    import open_clip
    # Same backbone, pretrained weights and corrected QuickGELU as PAR-CLIP.
    model = open_clip.create_model('ViT-B-32', pretrained='openai',
                                   device='cuda', force_quick_gelu=True)
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    unique = list(dict.fromkeys(texts1 + texts2))
    embs = []
    for ix in range(0, len(unique), 128):
        token_batch = tokenizer(unique[ix:ix+128]).cuda()
        with torch.autocast('cuda', dtype=torch.float16):
            v = model.encode_text(token_batch)
        embs.append(unit(v).cpu())
    table = torch.cat(embs)
    lookup = {text: i for i, text in enumerate(unique)}
    t1 = table[torch.tensor([lookup[x] for x in texts1])]
    t2 = table[torch.tensor([lookup[x] for x in texts2])]
    torch.save({'signature': signature, 't1': t1, 't2': t2}, cache)
    del model
    torch.cuda.empty_cache()
    print(f'Saved {attr} contextual text features, unique prompts={len(unique)}', flush=True)
    return t1, t2


class MLP(nn.Module):
    """Official Zhang MLP structure: 2 linear layers with ReLU including output."""
    def __init__(self, d, hidden):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                                     nn.Linear(hidden, d), nn.ReLU())

    def forward(self, x):
        return self.network(x)


class MatchedZhang(nn.Module):
    """Faithful modules/losses/queues on cached normalized CLIP features."""
    def __init__(self, d=512, queue_size=65536, alpha=0.3, tau=0.07):
        super().__init__()
        self.mlp1 = MLP(d, 512)
        self.mlp2 = MLP(d, 1024)
        self.mlp1_m = MLP(d, 512)
        self.mlp2_m = MLP(d, 1024)
        for p, q in zip(self.mlp1.parameters(), self.mlp1_m.parameters()):
            q.data.copy_(p.data)
            q.requires_grad_(False)
        for p, q in zip(self.mlp2.parameters(), self.mlp2_m.parameters()):
            q.data.copy_(p.data)
            q.requires_grad_(False)
        self.register_buffer('image_queue', F.normalize(torch.randn(d, queue_size), dim=0))
        self.register_buffer('text_queue', F.normalize(torch.randn(d, queue_size), dim=0))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        self.alpha, self.tau, self.momentum, self.ratio = alpha, tau, 0.995, 0.4

    def image(self, x):
        return F.normalize(self.mlp2(self.mlp1(x) + x) + x, dim=-1)

    def text(self, t):
        return F.normalize(self.mlp2(t) + t, dim=-1)

    @torch.no_grad()
    def momentum_update(self):
        for a, b in ((self.mlp1, self.mlp1_m), (self.mlp2, self.mlp2_m)):
            for p, q in zip(a.parameters(), b.parameters()):
                q.data.mul_(self.momentum).add_(p.data, alpha=1-self.momentum)

    @torch.no_grad()
    def enqueue(self, x, t):
        bs, capacity = len(x), self.image_queue.shape[1]
        ptr = int(self.queue_ptr.item())
        if ptr + bs <= capacity:
            self.image_queue[:,ptr:ptr+bs] = x.T
            self.text_queue[:,ptr:ptr+bs] = t.T
        else:
            first = capacity-ptr
            self.image_queue[:,ptr:] = x[:first].T
            self.text_queue[:,ptr:] = t[:first].T
            self.image_queue[:,:bs-first] = x[first:].T
            self.text_queue[:,:bs-first] = t[first:].T
        self.queue_ptr[0] = (ptr+bs)%capacity

    def forward_loss(self, img, t1, t2, epoch):
        # Canonical algorithm uses original features + momentum feature queues.
        oi, ot = F.normalize(img, dim=-1), F.normalize(t1, dim=-1)
        with torch.no_grad():
            self.momentum_update()
            im = F.normalize(self.mlp2_m(self.mlp1_m(img) + img) + img, dim=-1)
            tm = F.normalize(self.mlp2_m(t1) + t1, dim=-1)
            text_all = torch.cat((ot.T, self.text_queue.detach().clone()), 1)
            image_all = torch.cat((oi.T, self.image_queue.detach().clone()), 1)
            logits_it = oi @ text_all / self.tau
            logits_ti = ot @ image_all / self.tau
            target_it = F.softmax(logits_it, dim=-1)
            target_ti = F.softmax(logits_ti, dim=-1)
            if epoch >= 5:
                mit = F.softmax(im @ text_all / self.tau, dim=-1)
                mti = F.softmax(tm @ image_all / self.tau, dim=-1)
                target_it = (1-self.ratio)*target_it + self.ratio*mit
                target_ti = (1-self.ratio)*target_ti + self.ratio*mti
        fi = self.image(img)
        ft1, ft2 = self.text(t1), self.text(t2)
        biased_image = oi-fi
        biased_text = ot-ft1
        bi = biased_image @ image_all / self.tau
        bt = biased_text @ text_all / self.tau
        loss_ba = F.kl_div(F.log_softmax(bi, -1),
                           F.log_softmax(bt, -1),
                           reduction='batchmean', log_target=True)
        fair_text_random = ft1 if random.random() < .5 else ft2
        fair_it = fi @ text_all / self.tau
        fair_ti = fair_text_random @ image_all / self.tau
        loss_cd = ((-(F.log_softmax(fair_it, -1)*target_it).sum(-1).mean())
                   + (-(F.log_softmax(fair_ti, -1)*target_ti).sum(-1).mean()))/2
        with torch.no_grad():
            self.enqueue(oi, ot)
        return self.alpha*loss_cd + (1-self.alpha)*loss_ba


@torch.inference_mode()
def evaluate(model, images, labels, classes, occ, device):
    model.eval()
    res=[]
    for x in images.split(2048):
        res.append(model.image(x.to(device)).cpu())
    cc=model.text(classes.to(device)).cpu()
    oo=model.text(occ.to(device)).cpu()
    return scores(torch.cat(res), cc, oo, labels)


def run_attribute(attr, cfg):
    if cfg.get('embed_model') != 'ViT-B-32':
        raise ValueError('This runner only compares ViT-B-32')
    cache = ROOT / 'results' / f'cache_QuickGELU_{cfg["embed_model"]}_{attr}.pt'
    if not cache.exists():
        raise RuntimeError(f'Missing corrected PAR-CLIP embedding cache: {cache}')
    d = torch.load(cache, map_location='cpu', weights_only=True)
    frame = run.fairface_df(cfg, 'train', attr, cfg['train_cap'])
    if len(frame) != len(d['train_y']):
        raise RuntimeError('Dataset and cache row count mismatch')
    actual_labels = torch.tensor(frame['target'].to_numpy(), dtype=torch.long)
    if not torch.equal(actual_labels, d['train_y']):
        raise RuntimeError('FairFace training labels do not match cache order')

    t1,t2 = load_or_make_text_features(attr,cfg,frame)
    y = d['train_y'].cpu()
    tr_idx, va_idx = train_test_split(np.arange(len(y)), test_size=.2,
                                      random_state=int(cfg['seed']),
                                      stratify=y.numpy())
    tr_idx=torch.tensor(tr_idx,dtype=torch.long)
    va_idx=torch.tensor(va_idx,dtype=torch.long)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        raise RuntimeError('The matched comparison needs CUDA GPU')
    seed=int(cfg['seed'])
    set_seed(seed)
    model=MatchedZhang(d['train_x'].shape[1], alpha=.3,
                       tau=float(cfg['tau'])).to(device)
    optimizer=torch.optim.Adam(
        list(model.mlp1.parameters()) + list(model.mlp2.parameters()),
        lr=float(cfg['head_learning_rate']))
    generator=torch.Generator().manual_seed(seed)
    baseline_internal=scores(d['train_x'][va_idx], d['class_prompts'],
                             d['occupation_prompts'], y[va_idx])
    best=-float('inf'); best_state=None; best_epoch=0
    history=[]; start=0
    params={'attribute':attr,'backbone':'ViT-B-32','train_n':len(tr_idx),
            'internal_val_n':len(va_idx),'test_n':len(d['val_y']),
            'batch_size':int(cfg['head_batch_size']),
            'max_epochs':int(cfg['stage4_epochs']),
            'learning_rate':float(cfg['head_learning_rate']),
            'seed':seed,'tau':float(cfg['tau']),
            'zhang_alpha':.3,'momentum':.995,'queue_size':65536,
            'momentum_mix':.4,'evaluation':'same PAR-CLIP occupation DPG / demographic accuracy and macro-F1',
            'input':'cached normalized CLIP embeddings; Zhang official network/loss adapted to matched budget'}
    cp=HERE/f'{attr}_progress.pt'
    if cp.exists():
        old=torch.load(cp,map_location='cpu',weights_only=True)
        if old['params'] != params:
            raise RuntimeError(f'Checkpoint settings mismatch: {cp}')
        model.load_state_dict(old['model']);optimizer.load_state_dict(old['optimizer'])
        # Move CPU-loaded optimizer tensors to GPU after resume.
        for st in optimizer.state.values():
            for k,v in st.items():
                if isinstance(v,torch.Tensor):st[k]=v.to(device)
        generator.set_state(old['generator'])
        random.setstate(old['py_rng'])
        torch.set_rng_state(old['torch_rng'])
        torch.cuda.set_rng_state_all(old['cuda_rng'])
        best=old['best_score'];best_state=old['best_state']
        best_epoch=old['best_epoch'];history=old['history'];start=old['epoch']
        print(f'Resuming {attr} from epoch {start+1}',flush=True)
    train_x = d['train_x'][tr_idx].float()
    train_t1=t1[tr_idx].float(); train_t2=t2[tr_idx].float()
    for epoch in range(start,int(cfg['stage4_epochs'])):
        model.train();total=0.;steps=0
        for ids in torch.randperm(len(tr_idx),generator=generator).split(params['batch_size']):
            if len(ids)<2:continue
            image=train_x[ids].to(device)
            text1=train_t1[ids].to(device)
            text2=train_t2[ids].to(device)
            loss=model.forward_loss(image,text1,text2,epoch)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss for {attr} epoch {epoch+1}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total+=loss.detach().item();steps+=1
        internal=evaluate(model,d['train_x'][va_idx],y[va_idx],
                          d['class_prompts'],d['occupation_prompts'],device)
        f,dpg,a=internal['macro_f1'],internal['occupation_dpg'],internal['accuracy']
        sfu=2*f*(1-dpg)/max(f+1-dpg,1e-12)
        selection=sfu-2*max(0,baseline_internal['accuracy']-a)-2*max(0,baseline_internal['macro_f1']-f)
        history.append({'epoch':epoch+1,'train_loss':total/max(steps,1),
                        'internal_accuracy':a,'internal_macro_f1':f,
                        'internal_occupation_dpg':dpg,'selection_score':selection})
        if selection>best:
            best=selection;best_epoch=epoch+1
            # Store best model including text/image queues, so reloading faithfully reproduces predictions.
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        tmp=cp.with_name(cp.name+'.tmp')
        torch.save({'params':params,'epoch':epoch+1,
                    'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                    'optimizer':optimizer.state_dict(),'generator':generator.get_state(),
                    'py_rng':random.getstate(),'torch_rng':torch.get_rng_state(),
                    'cuda_rng':torch.cuda.get_rng_state_all(),
                    'best_score':best,'best_state':best_state,'best_epoch':best_epoch,
                    'history':history},tmp)
        tmp.replace(cp)
        csv_save(HERE/f'{attr}_history.csv',history)
        print(f'ZHANG-MATCHED {attr} epoch {epoch+1}/{params["max_epochs"]} '
              f'Acc={a:.4f} F1={f:.4f} DPG={dpg:.4f} '
              f'best_epoch={best_epoch}',flush=True)
    model.load_state_dict(best_state)
    model.eval()
    # Only now evaluate on the official held-out set (already studied in PAR-CLIP experiments).
    final=evaluate(model,d['val_x'],d['val_y'],d['class_prompts'],
                   d['occupation_prompts'],device)
    out={'attribute':attr,'method':'Zhang et al. matched-budget cached-feature adaptation',
         'backbone':'ViT-B-32','best_epoch':best_epoch,**final}
    torch.save({'model':best_state,'params':params,'best_epoch':best_epoch},
                HERE/f'{attr}_best.pt')
    json_save(HERE/f'{attr}_protocol.json',{
        **params,'best_epoch':best_epoch,'test_evaluated_after_internal_selection':True,
        'limitation':'Not exact published training: normalized cached CLIP inputs, matched budget and task-specific demographic evaluation. Original reported ABLE/MaxSkew are different metrics.'})
    json_save(HERE/f'{attr}_results.json',out)
    print('ZHANG-MATCHED FINAL',out,flush=True)
    return out


def self_test():
    set_seed(42)
    m=MatchedZhang(d=16,queue_size=64)
    b=4
    i=F.normalize(torch.randn(b,16),dim=-1)
    t=F.normalize(torch.randn(b,16),dim=-1)
    loss=m.forward_loss(i,t,t.roll(1,0),0)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.mlp1.network[0].weight.grad is not None
    with torch.no_grad():
        a=m.image(i)
    assert torch.allclose(a.norm(dim=-1),torch.ones(b),atol=1e-5)
    print('SELF TEST PASSED: Zhang model, queues, loss, gradients')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--attribute',choices=['age','race','both'],default='both')
    args=parser.parse_args()
    if args.self_test:
        self_test();return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU not available')
    if not (ROOT/'run.py').exists():
        raise RuntimeError('This file must live in ~/PAR_CLIP_Conference/zhang_reproduction/')
    cfg=run.config()
    if (int(cfg['train_cap']),int(cfg['head_batch_size']),int(cfg['stage4_epochs']),
        int(cfg['seed']),cfg['embed_model']) != (20000,256,30,42,'ViT-B-32'):
        raise ValueError('PAR-CLIP training settings changed; refuse to make misleading matched comparison')
    attrs=['age','race'] if args.attribute=='both' else [args.attribute]
    rows=[]
    for attr in attrs:
        resfile=HERE/f'{attr}_results.json'
        if resfile.exists():
            print('ALREADY COMPLETE',attr,'reading existing measured results',flush=True)
            rows.append(json.loads(resfile.read_text()))
            continue
        rows.append(run_attribute(attr,cfg))
    csv_save(HERE/'zhang_vitb32_matched_results.csv',rows)
    # One comparable table, using actual measured values only.
    compare = []
    baseline_csv = ROOT/'results'/'stage2_baseline.csv'
    if baseline_csv.exists():
        with baseline_csv.open(newline='') as f:
            for x in csv.DictReader(f):
                if x.get('attribute') in attrs:
                    compare.append({'attribute':x['attribute'], 'method':'Baseline CLIP',
                        'accuracy':x['accuracy'], 'macro_f1':x['macro_f1'],
                        'occupation_dpg':x['occupation_dpg']})
    age_report=ROOT/'results'/'age_retrain_evaluation.json'
    race_report=ROOT/'results'/'onepass_PARCLIP_holdout.csv'
    if 'age' in attrs and age_report.exists():
        age=json.loads(age_report.read_text())['official_candidate']
        compare.append({'attribute':'age','method':'PAR-CLIP age retrained',
                        **{k:age[k] for k in ('accuracy','macro_f1','occupation_dpg')}})
    if 'race' in attrs and race_report.exists():
        with race_report.open(newline='') as f:
            for x in csv.DictReader(f):
                if x.get('attribute')=='race':
                    compare.append({'attribute':'race','method':'PAR-CLIP race onepass',
                                    **{k:x[k] for k in ('accuracy','macro_f1','occupation_dpg')}})
                    break
    for x in rows:
        compare.append({'attribute':x['attribute'],'method':'Zhang matched-budget adaptation',
                        **{k:x[k] for k in ('accuracy','macro_f1','occupation_dpg')}})
    csv_save(HERE/'comparison_with_parclip.csv',compare)
    print('SAVED',HERE/'comparison_with_parclip.csv',flush=True)
    print('SAVED',HERE/'zhang_vitb32_matched_results.csv',flush=True)

if __name__=='__main__':
    main()
