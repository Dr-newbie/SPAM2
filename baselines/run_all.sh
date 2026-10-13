#!/usr/bin/env bash
# Run every baseline on one Xenium section under identical data and splits.
#
#   ./run_all.sh <h5ad> <patch_csv> <patch_root> <out_dir> [epochs] [gpu]
#
# Half precision is not used anywhere: fp16 and bf16 GEMM both fail on this
# machine's GPUs (see baselines/README.md).
set -euo pipefail

H5AD=${1:?h5ad}
CSV=${2:?patch csv}
ROOT=${3:?patch root}
OUT=${4:?output dir}
EPOCHS=${5:-250}
GPU=${6:-0}

BASE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH_PY=${SPAM_BENCH_PY:-python}
TF_PY=${SPAM_TF_PY:-python}
export SPAM_UNI_WEIGHTS=${SPAM_UNI_WEIGHTS:-/path/to/uni_weights}

mkdir -p "$OUT"

for M in stnet deepspace deeppt bleep stmcl; do
  echo "===================== $M"
  CUDA_VISIBLE_DEVICES=$GPU "$BENCH_PY" "$BASE/spam_bench/run_baseline.py" \
    --method "$M" --h5ad "$H5AD" --csv "$CSV" --root "$ROOT" \
    --epochs "$EPOCHS" --out "$OUT/$M" \
    2> "$OUT/$M.err" | tee "$OUT/$M.log"
done

echo "===================== egnv2"
CUDA_VISIBLE_DEVICES=$GPU "$BENCH_PY" "$BASE/spam_bench/run_egnv2.py" \
  --h5ad "$H5AD" --csv "$CSV" --root "$ROOT" \
  --epochs "$EPOCHS" --out "$OUT/egnv2" \
  2> "$OUT/egnv2.err" | tee "$OUT/egnv2.log"

echo "===================== stimage (tensorflow)"
CUDA_VISIBLE_DEVICES=$GPU "$TF_PY" "$BASE/spam_bench/run_stimage.py" \
  --h5ad "$H5AD" --csv "$CSV" --root "$ROOT" \
  --epochs "$EPOCHS" --out "$OUT/stimage" \
  2> "$OUT/stimage.err" | tee "$OUT/stimage.log"

echo
echo "===================== summary"
"$BENCH_PY" "$BASE/spam_bench/summarize.py" --runs "$OUT"
