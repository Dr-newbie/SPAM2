import warnings

import numpy as np
import torch
from torch_geometric.utils import add_self_loops, coalesce

try:
    from scipy.spatial import Delaunay, QhullError
    _HAVE_SCIPY = True
    _SCIPY_ERR = None
except Exception as e:
    Delaunay = None
    QhullError = Exception
    _HAVE_SCIPY = False
    _SCIPY_ERR = e


def _coalesce(edge_index: torch.Tensor,
              edge_attr: torch.Tensor | None,
              reduce: str = "min"):
    """
    torch_geometric 버전별 coalesce 시그니처 차이를 흡수.
    """
    try:
        return coalesce(edge_index, edge_attr, reduce=reduce)
    except TypeError:
        n = int(edge_index.max().item() + 1) if edge_index.numel() > 0 else 0
        return coalesce(edge_index, edge_attr, n, n, reduce=reduce)


def _knn_fallback(coords: np.ndarray, k: int):
    """
    scipy Delaunay를 쓸 수 없거나 실패할 때 사용할 KNN fallback.
    """
    n = int(coords.shape[0])
    if n <= 1:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

    k_eff = min(max(1, int(k)), n - 1)
    coords_t = torch.as_tensor(coords, dtype=torch.float32)
    k1 = k_eff + 1

    max_bytes = 256 * 1024 * 1024
    denom = max(1, n) * 4
    chunk = max(1, int(max_bytes / denom))
    chunk = min(chunk, 4096)

    dists_list = []
    idx_list = []
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        dist = torch.cdist(coords_t[s:e], coords_t)
        vals, idx = torch.topk(dist, k1, largest=False)
        dists_list.append(vals)
        idx_list.append(idx)

    dists = torch.cat(dists_list, dim=0).cpu().numpy()
    indices = torch.cat(idx_list, dim=0).cpu().numpy()

    rows = np.repeat(np.arange(n), k_eff)
    cols = indices[:, 1:k_eff + 1].reshape(-1)
    dist_vals = dists[:, 1:k_eff + 1].reshape(-1).astype(np.float32, copy=False)
    return rows.astype(np.int64, copy=False), cols.astype(np.int64, copy=False), dist_vals


@torch.no_grad()
def build_delaunay_graph(coords: np.ndarray,
                         k: int = 12,
                         use_weight: bool = True):
    """
    coords: (N,2) array of (x,y)
    k: Delaunay 실패 시 fallback KNN에서 사용할 이웃 수
    use_weight: 거리 기반 가우시안 edge weight 사용 여부

    Returns:
      edge_index: LongTensor (2, E)
      edge_weight: FloatTensor (E,) or None
    """
    assert coords.ndim == 2 and coords.shape[1] == 2, "coords must be (N,2)"
    n = int(coords.shape[0])

    if n == 0:
        return (torch.empty(2, 0, dtype=torch.long),
                None if not use_weight else torch.empty(0, dtype=torch.float32))

    coords = np.asarray(coords, dtype=np.float64)

    rows = np.empty(0, dtype=np.int64)
    cols = np.empty(0, dtype=np.int64)
    dist_vals = np.empty(0, dtype=np.float32)

    if n == 2:
        rows = np.array([0, 1], dtype=np.int64)
        cols = np.array([1, 0], dtype=np.int64)
        dist_vals = np.linalg.norm(coords[rows] - coords[cols], axis=1).astype(np.float32, copy=False)
    elif n >= 3 and _HAVE_SCIPY:
        try:
            tri = Delaunay(coords, qhull_options="QJ")
            simplices = tri.simplices
            if simplices.size > 0:
                e01 = simplices[:, [0, 1]]
                e12 = simplices[:, [1, 2]]
                e20 = simplices[:, [2, 0]]
                undirected = np.concatenate([e01, e12, e20], axis=0).astype(np.int64, copy=False)
                rows = undirected[:, 0]
                cols = undirected[:, 1]
                dist_vals = np.linalg.norm(coords[rows] - coords[cols], axis=1).astype(np.float32, copy=False)
        except QhullError:
            warnings.warn(
                "[build_delaunay_graph] Delaunay failed (degenerate geometry). "
                "Using KNN fallback."
            )
            rows, cols, dist_vals = _knn_fallback(coords, k=k)
    else:
        warnings.warn(
            "[build_delaunay_graph] scipy unavailable; using KNN fallback. "
            f"Original import error: {_SCIPY_ERR!r}"
        )
        rows, cols, dist_vals = _knn_fallback(coords, k=k)

    if rows.size > 0:
        row_all = np.concatenate([rows, cols], axis=0)
        col_all = np.concatenate([cols, rows], axis=0)
        dist_all = np.concatenate([dist_vals, dist_vals], axis=0)
        edge_index = torch.tensor(np.stack([row_all, col_all], axis=0), dtype=torch.long)
        edge_attr = torch.tensor(dist_all, dtype=torch.float32)
        edge_index, edge_attr = _coalesce(edge_index, edge_attr, reduce="min")
    else:
        edge_index = torch.empty(2, 0, dtype=torch.long)
        edge_attr = torch.empty(0, dtype=torch.float32)

    if use_weight:
        if edge_attr.numel() > 0:
            sigma = float(torch.median(edge_attr).item())
            if sigma <= 0:
                sigma = 1e-6
            edge_weight = torch.exp(-(edge_attr ** 2) / (2.0 * sigma ** 2))
            edge_index, edge_weight = add_self_loops(
                edge_index, edge_attr=edge_weight, num_nodes=n, fill_value=1.0
            )
        else:
            edge_index, edge_weight = add_self_loops(edge_index, num_nodes=n, fill_value=1.0)
    else:
        edge_index, edge_weight = add_self_loops(edge_index, num_nodes=n)

    return edge_index, edge_weight
