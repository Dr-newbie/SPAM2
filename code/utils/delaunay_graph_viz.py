"""
Delaunay Graph Visualization
Visualize Delaunay triangulation and KNN graphs for spatial coordinates.

Usage:


# Delaunay 그래프 시각화
python Code_final/code/utils/delaunay_graph_viz.py \
  --h5ad /data/project/banana9903_xenium/xenium_rep1_io/data/merged_with_celltype_raw.h5ad \
  --csv /data/project/banana9903_xenium/contrast_test/expression/breast1_patches_256/spot_patch_map_256.csv \
  --out ./viz_output \
  --graph_type delaunay \
  --k 12

# KNN vs Delaunay 비교
python Code_final/code/utils/delaunay_graph_viz.py \
  --h5ad /data/project/banana9903_xenium/xenium_rep1_io/data/merged_with_celltype_raw.h5ad \
  --csv /data/project/banana9903_xenium/contrast_test/expression/breast1_patches_256/spot_patch_map_256.csv \
  --out ./viz_output \
  --compare \
  --k 12 \
  --figsize 20 9


# with small samples
python Code_final/code/utils/delaunay_graph_viz.py \
  --h5ad /data/project/banana9903_xenium/xenium_rep1_io/data/merged_with_celltype_raw.h5ad \
  --csv /data/project/banana9903_xenium/contrast_test/expression/breast1_patches_256/spot_patch_map_256.csv \
  --out ./viz_output_sample \
  --compare --k 12 --sample 10000 --figsize 20 9

"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Tuple, Optional

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
from matplotlib.collections import LineCollection
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sys.modules.pop("code", None)

from code.utils.contrastive import build_spatial_graph
from code.utils.dataset_utils import (
    load_h5ad_with_preproc,
    load_patch_mapping_csv_xenium,
    align_map_ids_to_obs,
    reorder_and_prune_by_obs,
    extract_coords_np_in_order,
)


def plot_graph(
    coords: np.ndarray,
    edge_index: np.ndarray,
    edge_weight: Optional[np.ndarray] = None,
    title: str = "Spatial Graph",
    figsize: Tuple[int, int] = (12, 10),
    colorbar_label: str = "Edge Weight",
    dpi: int = 100,
) -> plt.Figure:
    """
    Plot spatial graph with nodes and edges.

    Args:
        coords: (N, 2) array of node coordinates
        edge_index: (2, E) edge indices
        edge_weight: (E,) edge weights (optional)
        title: plot title
        figsize: figure size
        colorbar_label: colorbar label
        dpi: figure DPI

    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot edges efficiently using LineCollection
    if edge_index.size > 0:
        src, dst = edge_index[0], edge_index[1]
        if len(src) > 0:
            # segments shape: (E, 2, 2)
            segments = np.stack([coords[src], coords[dst]], axis=1)
            if edge_weight is not None:
                # edge_weight should be (E,)
                lc = LineCollection(segments, array=edge_weight, cmap='viridis', linewidths=0.5, alpha=0.6)
                ax.add_collection(lc)
                fig.colorbar(lc, ax=ax, label=colorbar_label)
            else:
                lc = LineCollection(segments, colors='gray', linewidths=0.5, alpha=0.2)
                ax.add_collection(lc)
            ax.autoscale()

    # Plot nodes
    ax.scatter(
        coords[:, 0], coords[:, 1],
        s=50,
        c='red',
        alpha=0.8,
        edgecolors='darkred',
        linewidth=0.5,
        zorder=3,
        label='Nodes'
    )

    ax.set_xlabel('X coordinate (pixels)', fontsize=12)
    ax.set_ylabel('Y coordinate (pixels)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    return fig


def plot_graph_comparison(
    coords: np.ndarray,
    knn_edge_index: np.ndarray,
    delaunay_edge_index: np.ndarray,
    knn_edge_weight: Optional[np.ndarray] = None,
    delaunay_edge_weight: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (20, 9),
    dpi: int = 100,
) -> plt.Figure:
    """
    Plot KNN and Delaunay graphs side-by-side for comparison.

    Args:
        coords: (N, 2) coordinate array
        knn_edge_index: (2, E_knn) KNN edge indices
        delaunay_edge_index: (2, E_delaunay) Delaunay edge indices
        knn_edge_weight: (E_knn,) KNN edge weights
        delaunay_edge_weight: (E_delaunay,) Delaunay edge weights
        figsize: figure size
        dpi: figure DPI

    Returns:
        matplotlib figure with two subplots
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    # KNN plot
    ax = axes[0]
    if knn_edge_index.size > 0:
        src, dst = knn_edge_index[0], knn_edge_index[1]
        if len(src) > 0:
            segments = np.stack([coords[src], coords[dst]], axis=1)
            lc = LineCollection(segments, colors='blue', linewidths=0.5, alpha=0.2)
            ax.add_collection(lc)
            ax.autoscale()

    ax.scatter(
        coords[:, 0], coords[:, 1],
        s=1,
        c='red',
        alpha=0.5,
        edgecolors='darkred',
        linewidth=0.5,
        zorder=3,
    )
    ax.set_xlabel('X coordinate (pixels)', fontsize=11)
    ax.set_ylabel('Y coordinate (pixels)', fontsize=11)
    ax.set_title(f'KNN Graph (E={knn_edge_index.shape[1]})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal', adjustable='box')

    # Delaunay plot
    ax = axes[1]
    if delaunay_edge_index.size > 0:
        src, dst = delaunay_edge_index[0], delaunay_edge_index[1]
        if len(src) > 0:
            segments = np.stack([coords[src], coords[dst]], axis=1)
            lc = LineCollection(segments, colors='green', linewidths=0.5, alpha=0.3)
            ax.add_collection(lc)
            ax.autoscale()

    ax.scatter(
        coords[:, 0], coords[:, 1],
        s=1,
        c='red',
        alpha=0.5,
        edgecolors='darkred',
        linewidth=0.5,
        zorder=3,
    )
    ax.set_xlabel('X coordinate (pixels)', fontsize=11)
    ax.set_ylabel('Y coordinate (pixels)', fontsize=11)
    ax.set_title(f'Delaunay Graph (E={delaunay_edge_index.shape[1]})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser(description='Visualize spatial graphs (KNN vs Delaunay)')
    p.add_argument('--h5ad', required=True, help='H5AD file path')
    p.add_argument('--csv', required=True, help='CSV file path with spot_id and patch_path')
    p.add_argument('--root', default=None, help='Root directory for relative patch paths')
    p.add_argument('--graph_type', default='delaunay', choices=['knn', 'delaunay'],
                   help='Graph type to visualize')
    p.add_argument('--k', type=int, default=12, help='KNN k parameter')
    p.add_argument('--out', required=True, help='Output directory')
    p.add_argument('--coord_key', default='spatial', help='Coordinate key in h5ad')
    p.add_argument('--coord_norm', default='standardize', choices=['standardize', 'none'],
                   help='Coordinate normalization')
    p.add_argument('--sample', type=int, default=None,
                   help='Randomly sample N spots for visualization (speeds up plotting)')
    p.add_argument('--use_csv_coords', action='store_true', help='Use coordinates from CSV')
    p.add_argument('--csv_coord_cols', default='x_he_px,y_he_px', help='CSV coordinate columns')
    p.add_argument('--csv_to_unit', type=float, default=None, help='Scale factor for CSV coords')
    p.add_argument('--figsize', type=int, nargs=2, default=[12, 10], help='Figure size (width height)')
    p.add_argument('--dpi', type=int, default=100, help='Figure DPI')
    p.add_argument('--compare', action='store_true', help='Compare KNN and Delaunay graphs')
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"[Viz] Loading data from {args.h5ad}")

    # Load data
    adata = load_h5ad_with_preproc(
        args.h5ad,
        use_hvg=False,
        n_top_genes=None,
        x_key="x_centroid",
        y_key="y_centroid",
    )

    # Load mapping
    root_dir = args.root or os.path.dirname(args.csv)
    map_df = load_patch_mapping_csv_xenium(
        args.csv,
        root_dir=root_dir,
        drop_missing=True,
    )

    if len(map_df) == 0 and args.root is None:
        alt_root = os.path.dirname(root_dir)
        map_df = load_patch_mapping_csv_xenium(
            args.csv,
            root_dir=alt_root,
            drop_missing=True,
        )

    if len(map_df) == 0:
        raise RuntimeError(f"No valid patches found. Check --root and --csv paths.")

    map_df = align_map_ids_to_obs(adata, map_df, id_col="spot_id")
    map_df_aligned, coords_np = reorder_and_prune_by_obs(
        adata=adata, map_df=map_df, id_col="spot_id", coord_key=args.coord_key
    )

    if len(map_df_aligned) == 0:
        raise RuntimeError("No overlapping spot_ids. Check coordinate keys and CSV format.")

    if args.use_csv_coords:
        cx, cy = args.csv_coord_cols.split(",")
        coords_np = extract_coords_np_in_order(
            adata=adata,
            map_df=map_df_aligned,
            coord_key=args.coord_key,
            use_csv_coords=True,
            csv_coord_cols=(cx, cy),
            csv_to_unit=args.csv_to_unit,
        )

    # Normalize coordinates
    if args.coord_norm == "standardize":
        mean = coords_np.mean(axis=0, keepdims=True)
        std = coords_np.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        coords_viz = ((coords_np - mean) / std).astype(np.float32)
    else:
        coords_viz = coords_np.astype(np.float32)

    print(f"[Viz] Loaded {len(coords_np)} spots with coordinates shape {coords_np.shape}")

    # Build graphs
    if args.compare:
        print(f"[Viz] Building KNN and Delaunay graphs with k={args.k}")
        knn_ei, knn_ew = build_spatial_graph(coords_np, k=args.k, use_weight=True, graph_type='knn')
        delaunay_ei, delaunay_ew = build_spatial_graph(coords_np, k=args.k, use_weight=True, graph_type='delaunay')

        knn_ei = knn_ei.numpy()
        knn_ew = knn_ew.numpy() if knn_ew is not None else None
        delaunay_ei = delaunay_ei.numpy()
        delaunay_ew = delaunay_ew.numpy() if delaunay_ew is not None else None

        print(f"  KNN: {knn_ei.shape[1]} edges")
        print(f"  Delaunay: {delaunay_ei.shape[1]} edges")

        fig = plot_graph_comparison(
            coords_viz,
            knn_ei,
            delaunay_ei,
            knn_ew,
            delaunay_ew,
            figsize=tuple(args.figsize),
            dpi=args.dpi,
        )
        out_path = os.path.join(args.out, 'graph_comparison.png')
        fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
        print(f"[Viz] Saved comparison to {out_path}")

    else:
        print(f"[Viz] Building {args.graph_type} graph with k={args.k}")
        edge_index, edge_weight = build_spatial_graph(
            coords_np, k=args.k, use_weight=True, graph_type=args.graph_type
        )
        edge_index = edge_index.numpy()
        edge_weight = edge_weight.numpy() if edge_weight is not None else None

        print(f"  {args.graph_type.upper()}: {edge_index.shape[1]} edges")

        fig = plot_graph(
            coords_viz,
            edge_index,
            edge_weight,
            title=f'{args.graph_type.upper()} Graph (k={args.k})',
            figsize=tuple(args.figsize),
            dpi=args.dpi,
        )
        out_path = os.path.join(args.out, f'{args.graph_type}_graph.png')
        fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
        print(f"[Viz] Saved to {out_path}")

    plt.close('all')
    print("[Viz] Done!")


if __name__ == "__main__":
    main()
