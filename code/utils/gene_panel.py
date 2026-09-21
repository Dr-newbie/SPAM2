# -*- coding: utf-8 -*-
"""
Gene-panel handling shared by SPAM and the baseline comparison.

Xenium panels ship measured genes alongside quality-control probes that are
designed to produce no signal. The breast panel used here is 541 columns: 313
measured genes plus 228 controls —

    BLANK                159   unassigned codewords
    NegControlCodeword    41   codewords with no probe
    NegControlProbe       20   probes with no target
    antisense              8   antisense controls

They carry 0.070% of total counts (mean 0.0006 per cell vs 0.6234 for measured
genes). Left in, they contribute 228 near-unpredictable columns to any per-gene
average, so a "mean PCC over 541 genes" is not comparable to one over the
measured panel. They are also meaningless targets for a ZINB likelihood, which
models transcript counts.

Only the breast sections carry controls; the other panels here (377/382/480/
5001/5101) were already filtered upstream, so this is a no-op for them.

Both the SPAM pipeline and the baseline harness import this module, so the two
cannot drift apart — which is what makes "same genes, same metric" defensible.
"""
import re
from typing import Iterable, List

import numpy as np

# Case-insensitive prefixes for 10x Xenium control features.
CONTROL_PATTERN = re.compile(
    r"^(neg[_-]?control|negcontrol|blank|unassigned|deprecated|antisense|control)",
    re.IGNORECASE,
)


def is_control_probe(gene: str) -> bool:
    return bool(CONTROL_PATTERN.match(str(gene)))


def control_mask(gene_names: Iterable[str]) -> np.ndarray:
    """(G,) bool — True where the column is a control probe."""
    return np.fromiter((is_control_probe(g) for g in gene_names),
                       dtype=bool, count=len(list(gene_names)) if hasattr(gene_names, "__len__") else -1)


def measured_genes(gene_names: Iterable[str]) -> List[str]:
    return [g for g in gene_names if not is_control_probe(g)]


def drop_control_probes(adata, verbose: bool = True):
    """
    Return `adata` restricted to measured genes.

    Call this BEFORE library-size normalization so the size factor reflects
    transcript counts rather than counts plus control noise.
    """
    names = adata.var_names.astype(str)
    keep = np.array([not is_control_probe(g) for g in names], dtype=bool)
    n_drop = int((~keep).sum())
    if n_drop == 0:
        if verbose:
            print(f"[Panel] {adata.n_vars} genes, no control probes found")
        return adata
    if verbose:
        from collections import Counter
        kinds = Counter(re.match(r"^[A-Za-z]+", g).group(0)
                        for g in names[~keep] if re.match(r"^[A-Za-z]+", g))
        print(f"[Panel] dropped {n_drop} control probes "
              f"({dict(kinds)}); {int(keep.sum())} measured genes retained")
    return adata[:, keep].copy()
