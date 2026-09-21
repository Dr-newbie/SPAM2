# -*- coding: utf-8 -*-
"""
Spatial validation split for finetuning (reviewer comment R1-4).

The submitted implementation had no validation set at all: `train_epoch` and
`eval_epoch` were driven by the same dataloader, and the best checkpoint was
picked by Pearson correlation on the training section itself — in-sample model
selection, which tends to select the most overfit epoch rather than the best
one. The test section was never loaded during finetuning, so there was no test
leakage; the weakness was the absence of a held-out signal, not its
contamination.

Design constraints specific to this model:

1. **Not a random cell split.** Expression is strongly spatially autocorrelated,
   so a validation cell drawn at random sits among training cells and is far
   easier to predict than a genuinely held-out one.
2. **Not a single straight cut.** Halving a section biases composition — one
   half can be almost entirely stroma.
3. **The graph is not cut.** It is built over all N cells and only the *loss* is
   masked. This is standard transductive graph learning and keeps the tissue
   topology the model sees identical to inference.
4. **Buffer.** Cells within `buffer_um` of a tile boundary go to neither split,
   putting real distance between the two partitions.
5. **Every leakage path is closed downstream** — validation cells are excluded
   from the retrieval bank, from the Leiden clustering that defines the latent
   transcriptomic states, and from the state classifier's training set. These,
   not the convolution, are where a validation target could actually reach the
   model.

Sizing, measured on the breast section rather than assumed: Delaunay edges have
a median length of 34.7 um (p95 115.6 um), so two hops reach ~70 um typically.
Tiles of 1000 um are an order of magnitude larger than that and still give ~40
tiles across a 7.5 x 5.5 mm section, keeping composition balanced; a 75 um
buffer puts >=150 um between the nearest training and validation cells, about
four median hops.

One caveat worth stating plainly: because H(0) is a histology embedding
(comment R1-2), the graph propagates image features, not expression. No
validation *target* can reach a training node through the convolution. The
buffer therefore guards against the weaker effect that expression is spatially
autocorrelated, so a validation cell touching a training cell is easier to
predict than a genuinely held-out one.
"""
from typing import Tuple

import numpy as np


def checkerboard_split(
    coords: np.ndarray,
    val_frac: float = 0.2,
    tile_um: float = 1000.0,
    buffer_um: float = 75.0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Partition cells into spatially disjoint training and validation tiles.

    Args:
        coords: (N, 2) cell centroids, in the same units as tile_um (microns).
        val_frac: target fraction of non-buffer cells held out.
        tile_um: checkerboard tile edge length.
        buffer_um: cells this close to a tile edge are dropped from both splits.
        seed: controls which checkerboard tiles become validation.

    Returns:
        (train_idx, val_idx, buffer_idx) — disjoint, and their union is all cells.
    """
    xy = np.asarray(coords, dtype=np.float64)[:, :2]
    if xy.shape[0] == 0:
        e = np.empty(0, dtype=np.int64)
        return e, e, e

    origin = xy.min(axis=0)
    rel = xy - origin
    tx = np.floor(rel[:, 0] / tile_um).astype(np.int64)
    ty = np.floor(rel[:, 1] / tile_um).astype(np.int64)

    # Take validation tiles from one colour of the checkerboard, so held-out
    # tiles are never edge-adjacent to each other and stay spread over the
    # section instead of clumping.
    one_colour = ((tx + ty) % 2 == 1)
    tiles = np.unique(np.stack([tx, ty], axis=1)[one_colour], axis=0)
    if len(tiles) == 0:                       # section smaller than one tile
        one_colour = np.ones(len(xy), dtype=bool)
        tiles = np.unique(np.stack([tx, ty], axis=1), axis=0)

    rng = np.random.default_rng(seed)
    # tiles of one colour hold ~half the cells, so to hold out val_frac overall
    # we take ~2*val_frac of them
    n_pick = int(round(len(tiles) * min(1.0, 2.0 * val_frac)))
    n_pick = max(1, min(n_pick, len(tiles)))
    picked = tiles[rng.permutation(len(tiles))[:n_pick]]
    picked_set = set(map(tuple, picked.tolist()))

    is_val = np.fromiter(
        ((tx[i], ty[i]) in picked_set for i in range(len(xy))),
        dtype=bool, count=len(xy),
    )

    # distance to the nearest tile edge along each axis
    dx = rel[:, 0] % tile_um
    dy = rel[:, 1] % tile_um
    near_edge = (
        (np.minimum(dx, tile_um - dx) < buffer_um)
        | (np.minimum(dy, tile_um - dy) < buffer_um)
    )

    train_idx = np.where(~is_val & ~near_edge)[0]
    val_idx = np.where(is_val & ~near_edge)[0]
    buffer_idx = np.where(near_edge)[0]
    return train_idx, val_idx, buffer_idx


def summarize(coords: np.ndarray, train_idx, val_idx, buffer_idx,
              tile_um: float, buffer_um: float) -> str:
    n = len(coords)
    return (f"[ValSplit] checkerboard tile={tile_um:.0f}um buffer={buffer_um:.0f}um | "
            f"train={len(train_idx)} ({100*len(train_idx)/max(n,1):.1f}%) "
            f"val={len(val_idx)} ({100*len(val_idx)/max(n,1):.1f}%) "
            f"buffer={len(buffer_idx)} ({100*len(buffer_idx)/max(n,1):.1f}%)")
