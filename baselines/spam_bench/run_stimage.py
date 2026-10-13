#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STimage (Tan et al., Brief Bioinform 2023) on Xenium sections — TensorFlow.

STimage is the one baseline here that is not PyTorch, so it runs in its own
environment (.conda-envs/stimage_tf). We import `CNN_NB_multiple_genes` and
`negative_binomial_loss` from the upstream package unchanged: a ResNet50 trunk
with one negative-binomial head per gene, predicting E[y] = n*(1-p)/p.

Only the data source is replaced. Upstream reads a Visium AnnData plus tiles
written into `adata.obs['tile_path']`; we feed Xenium cell patches through a
tf.data pipeline with the same ResNet50 preprocessing.

Run with:
  .conda-envs/stimage_tf/bin/python run_stimage.py --h5ad ... --csv ... --out ...
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPOS = os.path.join(os.path.dirname(HERE), "repos")
sys.path.insert(0, os.path.join(REPOS, "STimage"))

from data import load_pair, spatial_block_split           # noqa: E402
from metrics import evaluate, save_predictions            # noqa: E402


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--h5ad", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--root", default=None)
    p.add_argument("--id_col", default="cell_id")
    p.add_argument("--path_col", default="patch_path")
    p.add_argument("--n_genes", type=int, default=None)
    p.add_argument("--test_h5ad", default=None)
    p.add_argument("--test_csv", default=None)
    p.add_argument("--test_root", default=None)
    p.add_argument("--inside_he_only", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--cnnbase", default="resnet50", choices=["resnet50", "densenet121"])
    p.add_argument("--finetune", action="store_true",
                   help="Unfreeze the CNN trunk (upstream `ft=True`).")
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def make_ds(paths, expr, idx, size, batch, shuffle, seed):
    import tensorflow as tf
    from tensorflow.keras.applications.resnet50 import preprocess_input

    p = tf.constant([paths[i] for i in idx])
    y = tf.constant(expr[idx], dtype=tf.float32)

    def load(path, target):
        img = tf.io.decode_png(tf.io.read_file(path), channels=3)
        img = tf.image.resize(img, (size, size))
        img = preprocess_input(tf.cast(img, tf.float32))
        # one NB head per gene -> the model wants a tuple of scalar targets
        return img, tuple(target[i] for i in range(target.shape[0]))

    ds = tf.data.Dataset.from_tensor_slices((p, y))
    if shuffle:
        ds = ds.shuffle(min(len(idx), 8192), seed=seed, reshuffle_each_iteration=True)
    return (ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(batch).prefetch(tf.data.AUTOTUNE))


def main():
    args = get_args()
    os.makedirs(args.out, exist_ok=True)

    import tensorflow as tf
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"[TF] {tf.__version__}, GPUs={len(tf.config.list_physical_devices('GPU'))}")

    from stimage._model import CNN_NB_multiple_genes, negative_binomial_loss

    tab, test_tab = load_pair(args, args.id_col, args.path_col)
    # STimage's NB heads model counts, not log1p(CP10K).
    expr = tab["counts"]
    n_cells, n_genes = expr.shape
    print(f"[Data] {n_cells} cells x {n_genes} genes (raw counts for the NB likelihood)")

    train_idx, val_idx = spatial_block_split(tab["coords"], args.val_frac, seed=args.seed)
    if args.limit:
        train_idx, val_idx = train_idx[:args.limit], val_idx[:max(1, args.limit // 4)]
    print(f"[Split] train={len(train_idx)} val={len(val_idx)}")

    model = CNN_NB_multiple_genes((args.img_size, args.img_size, 3), n_genes,
                                  cnnbase=args.cnnbase, ft=args.finetune)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                  loss=[negative_binomial_loss] * n_genes)
    print(f"[Model] STimage {args.cnnbase}: {model.count_params()/1e6:.2f}M params, "
          f"{n_genes} NB heads")

    tr = make_ds(tab["paths"], expr, train_idx, args.img_size, args.batch_size, True, args.seed)
    va = make_ds(tab["paths"], expr, val_idx, args.img_size, args.batch_size, False, args.seed)
    if test_tab is not None:
        eval_tab, eval_expr = test_tab, test_tab["counts"]
        eval_idx = np.arange(eval_expr.shape[0])
    else:
        eval_tab, eval_expr, eval_idx = tab, expr, val_idx
    ev = make_ds(eval_tab["paths"], eval_expr, eval_idx, args.img_size,
                 args.batch_size, False, args.seed)

    hist = model.fit(tr, epochs=args.epochs, validation_data=va, verbose=2)

    out = model.predict(ev, verbose=0)
    # each head emits (n, p) of the negative binomial; E[y] = n*(1-p)/p
    pred = np.stack([np.asarray(o)[:, 0] * (1 - np.asarray(o)[:, 1]) / np.asarray(o)[:, 1]
                     for o in out], axis=1)
    true = eval_expr[eval_idx]

    final = evaluate(pred, true, tab["gene_names"])
    final.update(method="stimage", epochs=args.epochs, n_genes=n_genes,
                 n_train=len(train_idx), n_val=len(eval_idx), cnnbase=args.cnnbase,
                 eval_set=("paired_test_section" if test_tab is not None
                           else "held_out_block"),
                 train_section=os.path.basename(args.h5ad),
                 test_section=(os.path.basename(args.test_h5ad)
                               if args.test_h5ad else os.path.basename(args.h5ad)))
    print(json.dumps(final, indent=2))

    save_predictions(os.path.join(args.out, "predictions.npz"), pred, true,
                     [eval_tab["ids"][i] for i in eval_idx], tab["gene_names"])
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"final": final,
                   "history": {k: [float(x) for x in v] for k, v in hist.history.items()}},
                  f, indent=2)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    model.save_weights(os.path.join(args.out, "model.weights.h5"))
    print(f"[Done] {args.out}")


if __name__ == "__main__":
    main()
