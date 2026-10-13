#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect every baseline's metrics.json into one comparison table."""
import argparse
import glob
import json
import os

import pandas as pd

COLS = ["method", "pcc_gene_mean", "pcc_gene_median", "pcc_gene_std",
        "pcc_top50heg_mean", "pcc_cell_mean", "mse", "mae",
        "n_genes_scored", "n_genes_undefined", "n_train", "n_val", "epochs"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="directory holding <method>/metrics.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.join(args.runs, "*", "metrics.json"))):
        with open(f) as fh:
            d = json.load(fh).get("final", {})
        d.setdefault("method", os.path.basename(os.path.dirname(f)))
        rows.append(d)
    if not rows:
        print(f"no metrics.json under {args.runs}")
        return

    df = pd.DataFrame(rows)
    df = df[[c for c in COLS if c in df.columns]]
    df = df.sort_values("pcc_gene_mean", ascending=False)
    pd.set_option("display.width", 200, "display.max_columns", 50)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    out = args.out or os.path.join(args.runs, "summary.csv")
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
