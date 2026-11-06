"""Small, explicit embedding methods for conference experiments (no journal code)."""
from __future__ import annotations
import copy
import math
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score


def unit(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1, eps=1e-8)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scores(images, classes, occupations, y):
    """Accuracy / macro-F1 classify demographics; DPG concerns neutral occupation probes.

    Important: demographic label is NOT the outcome for DPG. Prediction outcome
    is the top-1 matching occupation among fixed neutral probes.
    """
    x, p, o = unit(images).cpu(), unit(classes).cpu(), unit(occupations).cpu()
    y = torch.as_tensor(y, dtype=torch.long).cpu()
    pred = (x @ p.T).argmax(-1)
    valid = (y >= 0) & (y < len(p))
    if valid.sum() == 0:
        raise ValueError('No valid demographic labels for evaluation')
    acc = float(accuracy_score(y[valid].numpy(), pred[valid].numpy()))
    f1 = float(f1_score(y[valid].numpy(), pred[valid].numpy(), average='macro', zero_division=0))
    # Occupation-probe predictions are the Y-hat used in demographic parity.
    occ_pred = (x[valid] @ o.T).argmax(-1)
    groups = y[valid]
    gaps = []
    for c in range(len(o)):
        rates = [(occ_pred[groups == g] == c).float().mean().item()
                 for g in groups.unique() if (groups == g).any()]
        if len(rates) >= 2:
            gaps.append(max(rates) - min(rates))
    dpg = float(np.mean(gaps)) if gaps else 0.0
    counts = {str(int(g)): int((groups == g).sum()) for g in groups.unique()}
    return {'accuracy': round(acc, 6), 'macro_f1': round(f1, 6),
            'occupation_dpg': round(dpg, 6), 'n': int(valid.sum()),
            'groups': counts, 'n_occupation_probes': len(o)}


def subspace(z: torch.Tensor, labels: torch.Tensor, requested_k: int):
    """Group-mean SVD, using *training* data only."""
    z, labels = unit(z).cpu(), torch.as_tensor(labels).cpu()
    groups = sorted(int(g) for g in labels.unique() if int(g) >= 0)
    if len(groups) < 2:
        raise ValueError('At least two demographic groups needed for SVD')
    means = torch.stack([z[labels == g].mean(0) for g in groups])
    means_centered = means - means.mean(0, keepdim=True)
    _, singular, vt = torch.linalg.svd(means_centered, full_matrices=False)
    # After centering, rank <= (number of groups - 1).
    effective_k = min(int(requested_k), len(groups) - 1, vt.shape[0])
    if effective_k < 1:
        raise ValueError('k must be positive')
    b = vt[:effective_k].T.contiguous()
    return b, means, {'requested_k': int(requested_k),
                      'effective_k': effective_k, 'number_of_groups': len(groups),
                      'singular_values': singular[:effective_k].tolist()}


def remove(z, b):
    z, b = unit(z), b.float()
    return unit(z - (z @ b) @ b.T)


def procrustes(x, y, anchor_strength=1.0):
    """Orthogonal full-space map with identity anchoring for rank-deficient centroids.

    Both matrices are [N,D], representing paired (group-centroid, prompt) rows.
    """
    x, y = unit(x).cpu(), unit(y).cpu()
    d = x.shape[-1]
    if x.shape != y.shape:
        raise ValueError('Procrustes anchors need matched shapes')
    covariance = x.T @ y + float(anchor_strength) * torch.eye(d)
    u, _, vt = torch.linalg.svd(covariance, full_matrices=False)
    return u @ vt


class SmallResidualHead(nn.Module):
    """Only trainable parameters in PAR-CLIP; original CLIP stays frozen."""
    def __init__(self, d):
        super().__init__()
        self.linear = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        return unit(x + self.linear(x))


def parity_loss(logits, group_labels):
    """Differentiable class-average max/min group soft-rate difference."""
    prob = F.softmax(logits, dim=-1)
    unique = group_labels.unique()
    means = torch.stack([prob[group_labels == g].mean(0) for g in unique])
    return (means.max(0).values - means.min(0).values).mean()


def attenuate(z, basis, beta):
    """Partial projection for one-run internal-val-selected PAR-CLIP variant.

    beta=1 reproduces full (I-BB^T) projection; beta<1 is a disclosed variant.
    """
    z = unit(z).cpu()
    b = basis.float().cpu()
    return unit(z - float(beta) * (z @ b) @ b.T)


def evaluate_parclip_state(state, x, y, class_p, occ_p):
    """Apply *frozen*, training-fitted transforms to an unseen partition."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    feats = unit(attenuate(x, state['image_bias_basis'],
                           state['projection_strength']) @ state['procrustes_R'])
    head = SmallResidualHead(feats.shape[1]).to(device)
    head.load_state_dict(state['head_state_dict'])
    head.eval()
    outputs = []
    with torch.inference_mode():
        for batch in feats.split(2048):
            outputs.append(head(batch.to(device)).cpu())
    return scores(torch.cat(outputs), class_p, occ_p, y)


def train_parclip(train_x, train_y, val_x, val_y, class_p, occ_p,
                  requested_k, fairness_weight, epochs=30, batch_size=256,
                  lr=1e-4, tau=0.07, seed=42, anchor=1.0,
                  checkpoint_path=None, projection_strength=1.0):
    """Train-only SVD -> projection -> Procrustes -> supervised text InfoNCE + fairness.

    The train/val supplied here must both derive from the official FairFace
    TRAIN partition. Never pass official FairFace validation for selection.
    """
    set_seed(seed)
    if not (0 < float(projection_strength) <= 1):
        raise ValueError('projection_strength must be (0,1]')
    tr, va = unit(train_x).cpu(), unit(val_x).cpu()
    ty = torch.as_tensor(train_y).long().cpu()
    vy = torch.as_tensor(val_y).long().cpu()
    cls, occ = unit(class_p).cpu(), unit(occ_p).cpu()
    basis, _, meta = subspace(tr, ty, requested_k)
    meta['projection_strength'] = float(projection_strength)
    tr = attenuate(tr, basis, projection_strength)
    va = attenuate(va, basis, projection_strength)
    group_ids = sorted(int(i) for i in ty.unique().tolist())
    if group_ids != list(range(len(cls))):
        raise ValueError('Train demographic labels and text prompts do not match')
    centroids = torch.stack([tr[ty == g].mean(0) for g in group_ids])
    rotation = procrustes(centroids, cls, anchor)
    tr, va = unit(tr @ rotation), unit(va @ rotation)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    p, o = cls.to(device), occ.to(device)
    head = SmallResidualHead(tr.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.001)
    rng = torch.Generator().manual_seed(seed)
    # Selection penalizes utility loss rather than chasing artificially low DPG.
    raw_internal = scores(val_x, cls, occ, val_y)
    best_score, best_epoch, best = -float('inf'), 0, None
    history = []
    start_epoch = 0
    cp = Path(checkpoint_path) if checkpoint_path else None
    params = [int(requested_k), float(fairness_weight), int(epochs),
              int(batch_size), float(lr), float(tau), int(seed),
              float(anchor), float(projection_strength), len(ty), len(vy),
              int(ty.sum()), int(vy.sum()),
              round(float(train_x[:64].float().sum()), 5),
              round(float(val_x[:64].float().sum()), 5)]
    if cp is not None and cp.is_file():
        previous = torch.load(cp, map_location=device, weights_only=True)
        if previous['params'] != params:
            raise RuntimeError('Checkpoint/config mismatch: ' + str(cp))
        head.load_state_dict(previous['head'])
        opt.load_state_dict(previous['optimizer'])
        best = previous['best']
        best_epoch, best_score = previous['best_epoch'], previous['best_score']
        history, start_epoch = previous['history'], previous['epoch']
        rng.set_state(previous['rng'])
        print('Resuming', cp.name, 'at epoch', start_epoch + 1, flush=True)

    for epoch in range(start_epoch, int(epochs)):
        head.train()
        perm = torch.randperm(len(ty), generator=rng)
        total, count = 0., 0
        for ids in perm.split(int(batch_size)):
            if len(ids) < 4:
                continue
            z = head(tr[ids].to(device))
            labels = ty[ids].to(device)
            # Image -> correct demographic TEXT prompt is the supervised positive;
            # remaining fixed text prompts are negatives (class-prompt InfoNCE).
            info_nce = F.cross_entropy((z @ p.T) / tau, labels)
            fairness = parity_loss((z @ o.T) / tau, labels)
            loss = info_nce + float(fairness_weight) * fairness
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            count += 1
        head.eval()
        predictions = []
        with torch.inference_mode():
            for batch in va.split(2048):
                predictions.append(head(batch.to(device)).cpu())
        met = scores(torch.cat(predictions), cls, occ, vy)
        f, dpg, acc = met['macro_f1'], met['occupation_dpg'], met['accuracy']
        sfu = 2 * f * (1 - dpg) / max(f + 1 - dpg, 1e-12)
        # Strong penalty if model sacrifices demographic accuracy or macro-F1
        # against the unmodified CLIP *internal-validation* baseline.
        selection = (sfu - 2 * max(0, raw_internal['accuracy'] - acc)
                     - 2 * max(0, raw_internal['macro_f1'] - f))
        history.append({'epoch': epoch + 1, 'train_objective': total/max(count, 1),
                        'val_accuracy': acc, 'val_macro_f1': f,
                        'val_occupation_dpg': dpg, 'val_sfu': sfu,
                        'val_selection_score': selection})
        if selection > best_score:
            best_score, best_epoch = selection, epoch + 1
            best = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        if cp is not None:
            tmp = cp.with_suffix(cp.suffix + '.tmp')
            torch.save({'params': params, 'epoch': epoch+1, 'head': head.state_dict(),
                        'optimizer': opt.state_dict(), 'best': best, 'best_epoch':best_epoch,
                        'best_score':best_score, 'history': history, 'rng': rng.get_state()}, tmp)
            tmp.replace(cp)
        print(f'beta={projection_strength} epoch {epoch+1}/{epochs}: '
              f'acc={acc:.4f}, F1={f:.4f}, DPG={dpg:.4f}, selection={selection:.4f}',
              flush=True)

    if best is None:
        raise RuntimeError('No best checkpoint available')
    state = {'head_state_dict': best, 'image_bias_basis': basis,
             'procrustes_R': rotation, 'requested_k': int(requested_k),
             'effective_k': int(meta['effective_k']),
             'projection_strength': float(projection_strength),
             'lambda_fair': float(fairness_weight), 'tau': float(tau),
             'seed': int(seed), 'best_epoch':best_epoch}
    internal = evaluate_parclip_state(state, val_x, val_y, cls, occ)
    meta['best_epoch'] = best_epoch
    meta['best_val_selection_score'] = best_score
    return internal, meta, history, state


class ZhangStyleResidual(nn.Module):
    """Compact frozen-feature ADAPTATION of Zhang et al.; not their official reproduction.

    Two distinct residual transformations in the spirit of the CVPR 2025 method.
    Official method has momentum queues and more complex loss; this implementation
    must not be called an exact Zhang reproduction.
    """
    def __init__(self, d):
        super().__init__()
        self.image_head = nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Linear(512, d))
        self.both_head = nn.Sequential(nn.Linear(d, 1024), nn.ReLU(), nn.Linear(1024, d))
        # Near-identity start to preserve original cross-modal similarity.
        nn.init.zeros_(self.image_head[-1].weight)
        nn.init.zeros_(self.image_head[-1].bias)
        nn.init.zeros_(self.both_head[-1].weight)
        nn.init.zeros_(self.both_head[-1].bias)

    def image(self, z):
        return unit(z + self.both_head(z + self.image_head(z)))

    def text(self, z):
        return unit(z + self.both_head(z))


def train_zhang_style(train_x, train_y, val_x, val_y, class_p, occ_p,
                      epochs=5, batch_size=256, tau=0.07, seed=42):
    set_seed(seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_x, val_x, class_p, occ_p = map(lambda x: unit(x).cpu(),
                                         [train_x, val_x, class_p, occ_p])
    tr_y, va_y = torch.as_tensor(train_y).long(), torch.as_tensor(val_y).long()
    m = ZhangStyleResidual(train_x.shape[1]).to(dev)
    optimizer = torch.optim.AdamW(m.parameters(), lr=5e-5)
    original_p = class_p.to(dev)
    history = []
    rng = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        m.train(); total = 0.; steps = 0
        for ix in torch.randperm(len(tr_y), generator=rng).split(batch_size):
            if len(ix) < 4: continue
            x, y = train_x[ix].to(dev), tr_y[ix].to(dev)
            q_img = m.image(x)
            q_txt = m.text(original_p)
            supervised = F.cross_entropy((q_img @ q_txt.T) / tau, y)
            # Preserve original image-text geometry under adaptation.
            with torch.no_grad():
                t = F.softmax((x @ original_p.T) / tau, dim=1)
            alignment = F.kl_div(F.log_softmax((q_img @ q_txt.T) / tau, dim=1),
                                 t, reduction='batchmean')
            # Text counterfactual invariance: minimize variation under label swap.
            counter = q_txt.roll(1, dims=0)
            bias_penalty = (q_txt - counter).pow(2).mean()
            loss = 0.3 * supervised + 0.6 * alignment + 0.1 * bias_penalty
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total += loss.item(); steps += 1
        history.append({'epoch':epoch+1, 'training_objective':total/max(1,steps)})
    m.eval()
    with torch.no_grad():
        x = m.image(val_x.to(dev)).cpu()
        p = m.text(class_p.to(dev)).cpu()
        o = m.text(occ_p.to(dev)).cpu()
    metrics = scores(x, p, o, va_y)
    state = {k:v.cpu() for k,v in m.state_dict().items()}
    return metrics, history, state
