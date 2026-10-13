#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train and evaluate one baseline on a Xenium section (reviewer comment R2-1).

All methods see the same patches, the same gene panel, the same spatial split
and the same metrics; only the model and its training objective differ.

Two families:
  regression  (ST-Net, DeepSpaCE, DeepPT) — patch -> expression, MSE.
  contrastive (BLEEP, STMCL)              — patch/expression alignment, then
                                            prediction by retrieving the k
                                            nearest training spots in the joint
                                            embedding space and averaging their
                                            expression, as both papers specify.

Example
-------
python run_baseline.py --method stnet \
  --h5ad  /data/.../breast2_leiden.h5ad \
  --csv   /data/.../breast2_patches_256/spot_patch_map_256.csv \
  --root  /data/.../contrast_test/expression \
  --out   ../runs/stnet_breast2 --epochs 50
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import (PatchExprDataset, FeatureExprDataset, load_pair,
                  spatial_block_split)
from metrics import evaluate, save_predictions
import models as bench_models

CONTRASTIVE = {"bleep", "stmcl"}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True,
                   choices=["stnet", "deepspace", "bleep", "stmcl", "deeppt"])
    p.add_argument("--h5ad", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--root", default=None)
    p.add_argument("--id_col", default="cell_id")
    p.add_argument("--path_col", default="patch_path")
    p.add_argument("--test_h5ad", default=None,
                   help="Paired held-out section. If unset, the val block is scored.")
    p.add_argument("--test_csv", default=None)
    p.add_argument("--test_root", default=None)
    p.add_argument("--n_genes", type=int, default=None,
                   help="Keep the top-N genes by mean expression (shared across methods).")
    p.add_argument("--inside_he_only", action="store_true")

    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--knn", type=int, default=50,
                   help="Neighbours averaged at inference for contrastive methods.")
    p.add_argument("--limit", type=int, default=0, help="Debug: cap cell count.")
    p.add_argument("--uni_encoder", action="store_true",
                   help="R2 ablation: swap the image encoder for frozen UNI (uni_v1) "
                        "features (currently wired for deeppt).")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def make_loader(args, tab, idx, shuffle, feats=None):
    if feats is not None:
        ds = FeatureExprDataset(feats, tab["expr"], idx)
        nw = 2
    else:
        ds = PatchExprDataset(tab["paths"], tab["expr"], idx, size=args.img_size)
        nw = args.num_workers
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=nw, pin_memory=True, drop_last=False)


@torch.no_grad()
def extract_resnet_features(args, tab, tag="train"):
    """
    DeepPT stage 1: frozen ImageNet ResNet50 tile features.

    Upstream ships its own ResNet50_IMAGENET1K_V2.pt; we load it when present so
    the weights match the paper, otherwise torchvision's identical checkpoint.
    """
    import torchvision
    if getattr(args, "uni_encoder", False):
        # R2 fair-encoder ablation: frozen UNI (uni_v1) features instead of ResNet50.
        from uni_features import uni_features
        cache = os.path.join(args.out, f"uni_features_{tag}.npy")
        return uni_features(tab["paths"], cache, device=args.device,
                            batch_size=args.batch_size, num_workers=args.num_workers)
    cache = os.path.join(args.out, f"resnet50_features_{tag}.npy")
    if os.path.isfile(cache):
        return np.load(cache)

    local = os.path.join(bench_models.REPOS, "DeepPT", "ResNet50_IMAGENET1K_V2.pt")
    net = torchvision.models.resnet50(weights=None if os.path.isfile(local) else "IMAGENET1K_V2")
    if os.path.isfile(local):
        sd = torch.load(local, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        # Upstream stores the backbone under a "resnet." prefix.
        if all(k.startswith("resnet.") for k in sd):
            sd = {k[len("resnet."):]: v for k, v in sd.items()}
        missing, unexpected = net.load_state_dict(sd, strict=False)
        if any(not k.startswith("fc.") for k in missing) or unexpected:
            raise RuntimeError(
                f"DeepPT ResNet50 weights do not match torchvision resnet50: "
                f"missing={missing[:5]} unexpected={unexpected[:5]}")
        print(f"[DeepPT] loaded upstream ResNet50 weights from {local}")
    net.fc = nn.Identity()
    net = net.to(args.device).eval()

    ds = PatchExprDataset(tab["paths"], tab["expr"], size=args.img_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)
    out = np.zeros((len(ds), 2048), dtype=np.float32)
    from tqdm import tqdm
    for x, _, idx in tqdm(dl, desc="[DeepPT] ResNet50 features", leave=False):
        out[idx.numpy()] = net(x.to(args.device)).float().cpu().numpy()
    os.makedirs(args.out, exist_ok=True)
    np.save(cache, out)
    return out


def position_bins(coords, idx, n_bins=128):
    """STMCL indexes learned positional embeddings by integer bin."""
    xy = coords[idx][:, :2]
    lo, hi = xy.min(0), xy.max(0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    b = np.floor((xy - lo) / span * (n_bins - 1e-6)).astype(np.int64)
    return np.clip(b, 0, n_bins - 1)


def main():
    args = get_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tab, test_tab = load_pair(args, args.id_col, args.path_col)
    n_cells, n_genes = tab["expr"].shape
    print(f"[Data] train section: {n_cells} cells x {n_genes} genes")
    if test_tab is not None:
        print(f"[Data] test  section: {test_tab['expr'].shape[0]} cells "
              f"({n_genes} shared genes)")

    train_idx, val_idx = spatial_block_split(tab["coords"], args.val_frac, seed=args.seed)
    if args.limit:
        train_idx = train_idx[:args.limit]
        val_idx = val_idx[:max(1, args.limit // 4)]
    print(f"[Split] train={len(train_idx)} val={len(val_idx)} "
          f"(buffer dropped {n_cells - len(train_idx) - len(val_idx)})")

    # DeepPT trains on frozen CNN features, so they are extracted once per section.
    feats = extract_resnet_features(args, tab, "train") if args.method == "deeppt" else None
    test_feats = (extract_resnet_features(args, test_tab, "test")
                  if args.method == "deeppt" and test_tab is not None else None)
    feat_dim = feats.shape[1] if feats is not None else None

    train_dl = make_loader(args, tab, train_idx, True, feats)
    val_dl = make_loader(args, tab, val_idx, False, feats)
    # Scored set: the paired section when given, otherwise the held-out block.
    if test_tab is not None:
        eval_tab, eval_idx, eval_feats = test_tab, np.arange(test_tab["expr"].shape[0]), test_feats
    else:
        eval_tab, eval_idx, eval_feats = tab, val_idx, feats
    eval_dl = make_loader(args, eval_tab, eval_idx, False, eval_feats)

    if args.method == "deeppt":
        model = bench_models.build_deeppt(n_genes, n_inputs=feat_dim)
    else:
        model = bench_models.BUILDERS[args.method](n_genes)
    model = model.to(dev)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {args.method}: {n_par/1e6:.2f}M trainable params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    mse = nn.MSELoss()
    contrastive = args.method in CONTRASTIVE
    pos = position_bins(tab["coords"], np.arange(n_cells)) if args.method == "stmcl" else None
    eval_pos = (position_bins(eval_tab["coords"], np.arange(eval_tab["expr"].shape[0]))
                if args.method == "stmcl" else None)

    from tqdm import tqdm
    history = []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for batch in tqdm(train_dl, desc=f"[{args.method}] ep{ep}/{args.epochs}", leave=False):
            x, y, idx = batch
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if contrastive:
                d = {"image": x, "reduced_expression": y, "expression": y}
                if pos is not None:
                    d["position"] = torch.from_numpy(pos[idx.numpy()]).to(dev)
                loss = model(d)
                loss = loss.mean() if loss.dim() > 0 else loss
            else:
                loss = mse(model(x), y)
            loss.backward()
            opt.step()
            tot += float(loss.item()); nb += 1
        msg = f"[{ep:03d}] loss={tot/max(nb,1):.6f}"

        if ep % 10 == 0 or ep == args.epochs:
            pv, tv = predict(args, model, val_dl, tab, val_idx, tab, train_idx,
                             dev, contrastive, pos, pos)
            m = evaluate(pv, tv, tab["gene_names"])
            msg += (f" | val PCC(gene)={m['pcc_gene_mean']:.4f}"
                    f" med={m['pcc_gene_median']:.4f} MSE={m['mse']:.4e}")
            history.append({"epoch": ep, **m})
        print(msg, flush=True)

    pred, true = predict(args, model, eval_dl, eval_tab, eval_idx, tab, train_idx,
                         dev, contrastive, pos, eval_pos)
    final = evaluate(pred, true, tab["gene_names"])
    final.update(method=args.method, epochs=args.epochs, seconds=time.time() - t0,
                 n_train=len(train_idx), n_val=len(eval_idx), n_genes=n_genes,
                 eval_set=("paired_test_section" if test_tab is not None
                           else "held_out_block"),
                 train_section=os.path.basename(args.h5ad),
                 test_section=(os.path.basename(args.test_h5ad)
                               if args.test_h5ad else os.path.basename(args.h5ad)))
    print(json.dumps(final, indent=2))

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"final": final, "history": history}, f, indent=2)
    save_predictions(os.path.join(args.out, "predictions.npz"), pred, true,
                     [eval_tab["ids"][i] for i in eval_idx], tab["gene_names"])
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
    print(f"[Done] {args.out}")


@torch.no_grad()
def predict(args, model, eval_dl, eval_tab, eval_idx, bank_tab, bank_idx, dev,
            contrastive, bank_pos, eval_pos):
    """
    Score `eval_idx` of `eval_tab`. For the contrastive methods the retrieval
    bank is always the *training* cells of the training section, so a scored
    cell can never retrieve its own expression.
    """
    model.eval()
    if not contrastive:
        preds = []
        for x, _, _ in eval_dl:
            preds.append(model(x.to(dev)).float().cpu().numpy())
        return np.concatenate(preds, 0), eval_tab["expr"][eval_idx]

    def embed_images(dl):
        out = []
        for x, _, _ in dl:
            f = (model.image_encoder(x.to(dev)) if hasattr(model, "image_encoder")
                 else model.image_ecode(x.to(dev)))
            out.append(model.image_projection(f).float().cpu())
        return torch.cat(out, 0)

    def embed_spots(tab, idx, pos):
        e = torch.from_numpy(tab["expr"][idx]).to(dev)
        if pos is not None:
            b = torch.from_numpy(pos[idx]).to(dev)
            e = e + model.x_embed(b[:, 0]) + model.y_embed(b[:, 1])
        return model.spot_projection(e).float().cpu()

    q = torch.nn.functional.normalize(embed_images(eval_dl), dim=1)
    k = torch.nn.functional.normalize(embed_spots(bank_tab, bank_idx, bank_pos), dim=1)
    # Chunk the query rows so we never materialize the full (N_eval x N_bank)
    # similarity matrix — on tonsil that dense matrix is ~2.4 TB and OOMs. Do the
    # top-k on GPU per chunk when available, gather bank expression on CPU.
    kk = min(args.knn, k.shape[0])
    bank_expr = bank_tab["expr"][bank_idx]                     # (N_bank, G)
    sim_dev = dev if torch.cuda.is_available() else "cpu"
    kd = k.to(sim_dev)
    out = np.empty((q.shape[0], bank_expr.shape[1]), dtype=np.float32)
    CH = 2048
    for s in range(0, q.shape[0], CH):
        qb = q[s:s + CH].to(sim_dev)
        idx = (qb @ kd.t()).topk(kk, dim=1).indices.cpu().numpy()  # (chunk, kk)
        out[s:s + CH] = bank_expr[idx].mean(axis=1)
        del qb
    pred = out
    return pred, eval_tab["expr"][eval_idx]


if __name__ == "__main__":
    main()
