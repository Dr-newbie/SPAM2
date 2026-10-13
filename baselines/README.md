# Baseline benchmark harness

Unified pipeline used to train and evaluate every baseline reported in the SPAM
manuscript (ST-Net, DeepSpaCE, DeepPT, BLEEP, STMCL, EGNv2) on the six Xenium
cross-section transfer tasks, under identical data, splits, and evaluation.

Every run **saves its full configuration** to `<out>/<dataset>/<method>/config.json`
and its scores to `metrics.json`, so any reported number can be traced back to the
exact settings that produced it.

## Layout

```
baselines/
├── run_matrix.py        # orchestrator: all methods × all datasets (resumable)
├── run_all.sh           # all methods on ONE train/test pair
├── clone_repos.sh       # clones upstream implementations at pinned commits
├── datasets.json        # dataset registry (edit _root/_patch_root for your machine)
└── spam_bench/
    ├── run_baseline.py  # ST-Net / DeepSpaCE / DeepPT / BLEEP / STMCL
    ├── run_egnv2.py     # EGNv2 (exemplar-guided transformer)
    ├── run_stimage.py   # STimage (TensorFlow; not part of the reported benchmark)
    ├── models.py        # imports each architecture from the upstream repo
    ├── data.py          # shared Xenium loading (shared gene panel, patch CSVs)
    ├── metrics.py       # per-gene PCC / MSE / MAE on log1p(CP10K)
    ├── uni_features.py  # frozen-UNI encoder swap (UNI-backbone control, Table 5)
    └── summarize.py     # aggregates metrics.json files into one table
```

## Upstream implementations (pinned commits)

`bash clone_repos.sh` clones each repository into `baselines/repos/` at the commit
used for the manuscript:

| Method    | Repository                                   | Commit         | Date       |
|-----------|----------------------------------------------|----------------|------------|
| ST-Net    | github.com/bryanhe/ST-Net                    | `43022c1cb7de` | 2021-09-07 |
| DeepSpaCE | github.com/tmonjo/DeepSpaCE                  | `5e023eaa4df1` | 2022-10-30 |
| DeepPT    | authors' release; github.com/PangeaResearch/enlight-deeppt-data | `d96708890937` | 2024-03-12 |
| BLEEP     | github.com/bowang-lab/BLEEP                  | `23959670c740` | 2024-10-22 |
| STMCL     | github.com/wenwenmin/STMCL                   | `60843b2d1a8a` | 2024-10-25 |
| EGNv2     | github.com/Yan98/EGN                         | `bae8f0574d98` | 2023-10-20 |

Model architectures are imported from these repositories (`spam_bench/models.py`);
only the data loading and the training/evaluation loop are unified, so that every
method sees the same cells, the same shared gene panel, and the same splits.
DeepPT's training code is distributed by its authors as an archive, so its
three-stage pipeline (frozen ResNet50 features → autoencoder → MLP) is
re-implemented in `spam_bench/run_baseline.py`, optionally loading the authors'
released ResNet50 weights (verified weight-equivalent to torchvision's
`IMAGENET1K_V2`).

## Setup

```bash
pip install -r ../requirements.txt     # PyTorch env (see repository root)
bash clone_repos.sh                    # pinned upstream implementations

# optional: which python runs the jobs (defaults to the current python)
export SPAM_BENCH_PY=/path/to/torch-env/bin/python
export SPAM_TF_PY=/path/to/tf-env/bin/python      # only needed for STimage
export SPAM_UNI_WEIGHTS=/path/to/uni_weights      # only for --uni_encoder control
```

Then edit `datasets.json`: set `"_root"` to the directory that contains your
`h5ads/` and `contrast_test/expression/` patch folders (per-dataset paths inside
the file are relative to `_root`).

## Running

**Full benchmark matrix** (all methods × all datasets; resumable — finished jobs
with a `metrics.json` are skipped):

```bash
python run_matrix.py --out runs/matrix --gpus 0,1 --parallel 8
```

**One train/test pair, all methods:**

```bash
./run_all.sh <train.h5ad> <train_patch_map.csv> <patch_root> runs/single [epochs] [gpu]
```

**A single method:**

```bash
python spam_bench/run_baseline.py --method stnet \
  --h5ad <train.h5ad> --csv <train_patch_map.csv> --root <patch_root> \
  --test_h5ad <test.h5ad> --test_csv <test_patch_map.csv> --test_root <patch_root> \
  --out runs/stnet_breast
```

**UNI-backbone control** (Table 5 — every supported baseline gets SPAM's frozen
UNI encoder):

```bash
python run_matrix.py --out runs/matrix_uni --uni_encoder
```

**Aggregate results:**

```bash
python spam_bench/summarize.py --runs runs/matrix
```

## Configuration

Defaults (also recorded per run in `config.json`): batch size 128, learning rate
1e-4, weight decay 1e-4, seed 0, 224×224 patches, 20% held-out validation split
inside the training section; EGNv2 uses batch size 4 and learning rate 5e-4
(exemplar-transformer memory footprint). Mixed precision is disabled everywhere.
All metrics are computed per gene on log1p(CP10K) over the shared measured-gene
panel (negative-control probes removed), identically for every method.

The exact configuration used for each number reported in the manuscript is the
`config.json` saved next to that run's `metrics.json` (see Supplementary
Table for the per-method summary).
