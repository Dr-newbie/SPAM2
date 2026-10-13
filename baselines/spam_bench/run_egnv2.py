#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EGNv2 / EGGN (Yang et al., Pattern Recognition 2024) on Xenium sections.

The upstream pipeline is three scripts that assume a Visium tree (read_visium +
a whole-slide .tif) and drives training with pytorch_lightning 1.5.7, whose API
(`gpus=`, `DDPPlugin`, `checkpoint_callback=`) no longer exists in current
Lightning. What we keep is the part that *is* the method:

  * `model.HeteroGNN` — the heterogeneous exemplar/window GNN, imported verbatim,
    together with its `EGBBlock`, `HeteroConv` and `CSRA` pooling.
  * The graph construction recipe from `generate_graph.py`: a radius graph over
    window patches, a kNN graph over exemplars, and cross edges from exemplars
    to the windows that retrieved them.
  * The objective from `train.py`: MSE + 0.5 * (1 - correlation).

What we replace is the data source (Xenium cells instead of Visium spots) and
the trainer (a plain loop). `out_channels` is rebound from the hard-coded 250 to
our panel size.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import PatchExprDataset, load_pair, spatial_block_split
from metrics import evaluate, save_predictions
import models as bench_models

EGN_V2 = os.path.join(bench_models.REPOS, "EGN", "v2")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--h5ad", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--root", default=None)
    p.add_argument("--id_col", default="cell_id")
    p.add_argument("--path_col", default="patch_path")
    p.add_argument("--n_genes", type=int, default=None)
    p.add_argument("--test_h5ad", default=None,
                   help="Paired held-out section. If unset, the val block is scored.")
    p.add_argument("--test_csv", default=None)
    p.add_argument("--test_root", default=None)
    p.add_argument("--inside_he_only", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--mdim", type=int, default=512)
    p.add_argument("--numk", type=int, default=6, help="exemplars retrieved per window")
    p.add_argument("--window", type=int, default=512, help="cells per window graph")
    p.add_argument("--radius", type=float, default=120.0, help="window-window edge radius (um)")
    p.add_argument("--batch_size", type=int, default=4, help="window graphs per step")
    p.add_argument("--feat_batch", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--uni_encoder", action="store_true",
                   help="R2 ablation: use frozen UNI (uni_v1) patch features instead of ResNet18.")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


@torch.no_grad()
def patch_features(args, tab, dev, tag="train"):
    """
    Exemplar features. build_exemplar.py uses an ImageNet ResNet with the head
    stripped; we mirror that and project to --mdim, which is what HeteroGNN
    expects as `input_channel`.
    """
    cache = os.path.join(args.out, f"egn_feats_{args.mdim}_{tag}.npy")
    if os.path.isfile(cache):
        return np.load(cache)
    if getattr(args, "uni_encoder", False):
        # R2 fair-encoder ablation: frozen UNI (uni_v1) features (1024-d) then ->mdim.
        from uni_features import uni_features
        ucache = os.path.join(args.out, f"uni_raw_{tag}.npy")
        out = uni_features(tab["paths"], ucache, device=dev,
                           batch_size=args.feat_batch, num_workers=args.num_workers)
        feat_dim = out.shape[1]
    else:
        import torchvision
        enc = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        feat_dim = enc.fc.in_features
        enc = nn.Sequential(*list(enc.children())[:-1]).to(dev).eval()

        ds = PatchExprDataset(tab["paths"], tab["expr"], size=224)
        dl = torch.utils.data.DataLoader(ds, batch_size=args.feat_batch, shuffle=False,
                                         num_workers=args.num_workers, pin_memory=True)
        out = np.zeros((len(ds), feat_dim), dtype=np.float32)
        from tqdm import tqdm
        for x, _, idx in tqdm(dl, desc="[EGNv2] patch features", leave=False):
            out[idx.numpy()] = enc(x.to(dev)).flatten(1).float().cpu().numpy()
    if feat_dim != args.mdim:
        rng = np.random.default_rng(args.seed)
        proj = rng.normal(0, 1 / np.sqrt(feat_dim), (feat_dim, args.mdim)).astype(np.float32)
        out = out @ proj
    os.makedirs(args.out, exist_ok=True)
    np.save(cache, out)
    return out


def build_windows(coords, idx, window):
    """Partition cells into spatially contiguous windows (upstream slices a slide)."""
    order = idx[np.lexsort((coords[idx, 1], coords[idx, 0]))]
    return [order[i:i + window] for i in range(0, len(order), window)
            if len(order[i:i + window]) >= 8]


def radius_graph_t(pos: torch.Tensor, r: float) -> torch.Tensor:
    """
    Dense radius graph. torch_geometric's version needs torch-cluster, which has
    no wheel for this torch build; windows are <=512 nodes so an O(n^2) distance
    matrix is cheaper than compiling it.
    """
    d = torch.cdist(pos, pos)
    mask = (d <= r)
    mask.fill_diagonal_(False)
    src, dst = mask.nonzero(as_tuple=True)
    return torch.stack([src, dst])


def knn_graph_t(x: torch.Tensor, k: int) -> torch.Tensor:
    """Dense kNN graph (self excluded), same reason as above."""
    n = x.shape[0]
    k = min(k, max(n - 1, 1))
    d = torch.cdist(x, x)
    d.fill_diagonal_(float("inf"))
    idx = d.topk(k, largest=False).indices
    src = torch.arange(n).repeat_interleave(k)
    return torch.stack([src, idx.reshape(-1)])


def make_graph(win_idx, feats, expr, coords, ex_idx, ex_feats, ex_expr,
               numk, radius, HeteroData, radius_graph, knn_graph):
    """One HeteroData window graph, following generate_graph.py."""
    import torch as T
    w_feat = T.from_numpy(feats[win_idx])
    w_y = T.from_numpy(expr[win_idx])
    w_pos = T.from_numpy(coords[win_idx][:, :2].astype(np.float32))

    # exemplars: numk nearest training patches in feature space
    q = F.normalize(w_feat, dim=1)
    k = F.normalize(T.from_numpy(ex_feats), dim=1)
    top = (q @ k.t()).topk(min(numk, k.shape[0]), dim=1).indices
    flat = top.reshape(-1)
    e_feat = T.from_numpy(ex_feats[flat.numpy()])
    e_y = T.from_numpy(ex_expr[flat.numpy()])

    d = HeteroData()
    d["window"].x = w_feat
    d["window"].y = w_y
    # upstream concatenates the exemplar's expression onto its feature
    d["example"].x = T.cat([e_feat, e_y], dim=1)

    d["window", "near", "window"].edge_index = radius_graph(w_pos, radius)
    src = T.arange(len(win_idx)).repeat_interleave(top.shape[1])
    dst = T.arange(flat.numel())
    d["example", "refer", "window"].edge_index = T.stack([dst, src])
    d["example", "close", "example"].edge_index = knn_graph(e_feat, 3)
    return d


def correlation_metric(pred, true, eps=1e-8):
    """1 - mean Pearson r across genes; the auxiliary term in train.py."""
    p = pred - pred.mean(0, keepdim=True)
    t = true - true.mean(0, keepdim=True)
    r = (p * t).sum(0) / (p.norm(dim=0) * t.norm(dim=0) + eps)
    return 1.0 - r.mean()


def main():
    args = get_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    from torch_geometric.data import HeteroData
    from torch_geometric.loader import DataLoader as GeoLoader
    knn_graph, radius_graph = knn_graph_t, radius_graph_t

    tab, test_tab = load_pair(args, args.id_col, args.path_col)
    n_cells, n_genes = tab["expr"].shape
    print(f"[Data] train section: {n_cells} cells x {n_genes} genes")
    if test_tab is not None:
        print(f"[Data] test  section: {test_tab['expr'].shape[0]} cells")

    train_idx, val_idx = spatial_block_split(tab["coords"], args.val_frac, seed=args.seed)
    if args.limit:
        train_idx, val_idx = train_idx[:args.limit], val_idx[:max(1, args.limit // 4)]
    print(f"[Split] train={len(train_idx)} val={len(val_idx)}")

    feats = patch_features(args, tab, dev, "train")
    test_feats = (patch_features(args, test_tab, dev, "test")
                  if test_tab is not None else None)

    # Exemplar bank = training cells only, so validation windows cannot retrieve
    # their own expression.
    ex_feats, ex_expr = feats[train_idx], tab["expr"][train_idx]

    def graphs_for(src_tab, src_feats, idx):
        # exemplars always come from the training section's training cells
        return [make_graph(w, src_feats, src_tab["expr"], src_tab["coords"], train_idx,
                           ex_feats, ex_expr, args.numk, args.radius,
                           HeteroData, radius_graph, knn_graph)
                for w in build_windows(src_tab["coords"], idx, args.window)]

    t0 = time.time()
    train_g = graphs_for(tab, feats, train_idx)
    val_g = graphs_for(tab, feats, val_idx)
    if test_tab is not None:
        eval_g = graphs_for(test_tab, test_feats,
                            np.arange(test_tab["expr"].shape[0]))
    else:
        eval_g = val_g
    print(f"[Graphs] train={len(train_g)} val={len(val_g)} eval={len(eval_g)} "
          f"windows ({time.time()-t0:.1f}s)")

    sys.path.insert(0, EGN_V2)
    from model import HeteroGNN
    model = HeteroGNN(args.num_layers, args.mdim)
    # upstream hard-codes out_channels=250 for its Visium panel
    import torch_geometric.nn as pyg
    model.lin = pyg.Linear(512, n_genes)
    model.pretransform_exp = pyg.Linear(args.mdim + n_genes, 512, bias=False)
    model.pretransform_ey = pyg.Linear(n_genes, 512, bias=False)
    model._egn_n_genes = n_genes
    model = model.to(dev)
    print(f"[Model] EGNv2 HeteroGNN: "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    # forward() slices the last 250 columns for example_y; rebind to n_genes
    orig_forward = HeteroGNN.forward

    def forward(self, x_dict, edge_index_dict):
        g = self._egn_n_genes
        x_dict = dict(x_dict)
        ex_y = x_dict["example"][:, -g:]
        x_dict["example"] = self.post_transform(self.pretransform_exp(x_dict["example"]))
        x_dict["window"] = self.post_transform(self.pretransform_win(x_dict["window"]))
        x_dict["example_y"] = self.pretransform_ey(ex_y)
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: self.leaklyrelu(v) for k, v in x_dict.items()}
        return self.lin(self.pool(x_dict, edge_index_dict))

    HeteroGNN.forward = forward

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    train_dl = GeoLoader(train_g, batch_size=args.batch_size, shuffle=True)
    val_dl = GeoLoader(val_g, batch_size=args.batch_size, shuffle=False)
    eval_dl = GeoLoader(eval_g, batch_size=args.batch_size, shuffle=False)

    from tqdm import tqdm
    history = []
    for ep in range(1, args.epochs + 1):
        model.train(); tot = nb = 0
        for d in tqdm(train_dl, desc=f"[egnv2] ep{ep}/{args.epochs}", leave=False):
            d = d.to(dev)
            opt.zero_grad(set_to_none=True)
            pred = model(d.x_dict, d.edge_index_dict)
            y = d["window"].y
            loss = F.mse_loss(pred, y) + 0.5 * correlation_metric(pred, y)
            loss.backward(); opt.step()
            tot += float(loss.item()); nb += 1
        msg = f"[{ep:03d}] loss={tot/max(nb,1):.6f}"
        if ep % 5 == 0 or ep == args.epochs:
            pred, true = predict(model, val_dl, dev)
            m = evaluate(pred, true, tab["gene_names"])
            msg += f" | val PCC(gene)={m['pcc_gene_mean']:.4f} MSE={m['mse']:.4e}"
            history.append({"epoch": ep, **m})
        print(msg, flush=True)

    pred, true = predict(model, eval_dl, dev)
    final = evaluate(pred, true, tab["gene_names"])
    final.update(method="egnv2", epochs=args.epochs, n_genes=n_genes,
                 n_train=len(train_idx), n_val=len(true),
                 eval_set=("paired_test_section" if test_tab is not None
                           else "held_out_block"),
                 train_section=os.path.basename(args.h5ad),
                 test_section=(os.path.basename(args.test_h5ad)
                               if args.test_h5ad else os.path.basename(args.h5ad)))
    print(json.dumps(final, indent=2))
    save_predictions(os.path.join(args.out, "predictions.npz"), pred, true,
                     [""] * len(pred), tab["gene_names"])
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"final": final, "history": history}, f, indent=2)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
    print(f"[Done] {args.out}")


@torch.no_grad()
def predict(model, dl, dev):
    model.eval()
    P, Y = [], []
    for d in dl:
        d = d.to(dev)
        P.append(model(d.x_dict, d.edge_index_dict).float().cpu().numpy())
        Y.append(d["window"].y.float().cpu().numpy())
    return np.concatenate(P, 0), np.concatenate(Y, 0)


if __name__ == "__main__":
    main()
