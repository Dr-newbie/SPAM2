#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run every baseline over every paired dataset, sequentially, on one GPU.

Each job trains on a section and scores the paired section (datasets.json).
Jobs are skipped when metrics.json already exists, so the matrix is resumable
after an interruption.

  python run_matrix.py --out runs/matrix --epochs 100 --gpu 0
  python run_matrix.py --out runs/matrix --epochs 100 --gpu 1 \
      --datasets skin skin_cancer --methods stnet deepspace
"""
import argparse
import json
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
BENCH_PY = os.environ.get("SPAM_BENCH_PY", sys.executable)  # python with torch env (see README)
TF_PY = os.environ.get("SPAM_TF_PY", sys.executable)        # tensorflow env, only needed for STimage
ALL_METHODS = ["stnet", "deepspace", "deeppt", "bleep", "stmcl", "egnv2", "stimage"]


def job_cmd(method, ds, cfg, out, args, root):
    tr, te = cfg["train"], cfg["test"]
    common = ["--h5ad", os.path.join(root, tr["h5ad"]),
              "--csv", os.path.join(root, tr["csv"]),
              "--root", os.path.join(root, "contrast_test/expression"),
              "--epochs", str(args.epochs), "--out", out]
    if te:
        common += ["--test_h5ad", os.path.join(root, te["h5ad"]),
                   "--test_csv", os.path.join(root, te["csv"]),
                   "--test_root", os.path.join(root, "contrast_test/expression")]
    if args.limit:
        common += ["--limit", str(args.limit)]

    # R2 fair-encoder ablation: swap image encoder for frozen UNI. Currently wired
    # for deeppt and egnv2 (clean frozen-feature baselines).
    uni = getattr(args, "uni_encoder", False)
    UNI_SUPPORTED = {"deeppt", "egnv2"}
    uni_flag = ["--uni_encoder"] if (uni and method in UNI_SUPPORTED) else []

    if method == "stimage":
        return [TF_PY, os.path.join(BASE, "spam_bench/run_stimage.py")] + common
    if method == "egnv2":
        return [BENCH_PY, os.path.join(BASE, "spam_bench/run_egnv2.py")] + common + \
               ["--num_workers", str(args.num_workers)] + uni_flag
    return [BENCH_PY, os.path.join(BASE, "spam_bench/run_baseline.py"),
            "--method", method] + common + \
           ["--num_workers", str(args.num_workers), "--batch_size", str(args.batch_size)] + uni_flag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--gpus", default="0,1",
                    help="Comma-separated GPU ids; jobs are dealt round-robin.")
    ap.add_argument("--parallel", type=int, default=8,
                    help="Concurrent jobs in total, spread over --gpus.")
    ap.add_argument("--methods", nargs="*", default=ALL_METHODS)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--uni_encoder", action="store_true",
                    help="R2 ablation: swap baseline image encoders for frozen UNI "
                         "(currently deeppt, egnv2). Use a separate --out dir.")
    ap.add_argument("--min_free_mb", type=int, default=24000,
                    help="Launch on a GPU only if it has at least this much free "
                         "memory - respects jobs owned by other schedulers.")
    ap.add_argument("--spam_guard", action="store_true",
                    help="Never launch while a SPAM process (pretraining, "
                         "finetuning, inference) is running anywhere. SPAM owns "
                         "GPU 0 by convention; use this on the GPU-0 scheduler.")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    with open(os.path.join(BASE, "datasets.json")) as f:
        manifest = json.load(f)
    root = manifest["_root"]
    datasets = manifest["datasets"]
    names = args.datasets or list(datasets)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    os.makedirs(args.out, exist_ok=True)

    # Interleave methods across datasets so concurrent jobs mostly touch
    # different sections; identical-dataset jobs would otherwise all load the
    # same multi-GB expression matrix at once.
    pending = []
    for m_i, method in enumerate(args.methods):
        for d_i, dname in enumerate(names):
            pending.append((dname, method))
    pending.sort(key=lambda t: (args.methods.index(t[1]) + names.index(t[0])))

    todo = []
    for dname, method in pending:
        out = os.path.join(args.out, dname, method)
        if os.path.isfile(os.path.join(out, "metrics.json")):
            print(f"skip {dname}/{method} (already done)")
            continue
        todo.append((dname, method, out))

    print(f"{len(todo)} jobs, {args.parallel} at a time, GPUs {gpus}, "
          f"{args.num_workers} loader workers each\n", flush=True)

    if args.dry_run:
        for i, (d, m, out) in enumerate(todo):
            print(f"[{i+1}] gpu{gpus[i % len(gpus)]} {d}/{m}")
        return

    running = []          # (proc, dname, method, gpu, t0, logs)
    launched = 0
    done_n = 0
    total = len(todo)

    def claimed_elsewhere(out):
        """
        True if the job is finished or currently owned by another scheduler.

        The todo list is built once at startup, so with two schedulers running
        (one per GPU) each must re-check at launch time or both will run the
        same job. Ownership is detected from the --out path in a live process's
        command line; the window between this check and our own Popen is
        milliseconds, acceptable for this use.
        """
        if os.path.isfile(os.path.join(out, "metrics.json")):
            return True
        # pid-file ownership: the pgrep-on-cmdline check failed in practice
        # (tonsil/stnet and tonsil/deepspace were double-launched, and the
        # duplicate's "w" open truncated the original's run.log).
        pf = os.path.join(out, "started.pid")
        if os.path.isfile(pf):
            try:
                pid = int(open(pf).read().strip())
                os.kill(pid, 0)
                return True
            except (ValueError, ProcessLookupError, PermissionError):
                pass
        return False

    SPAM_PATTERN = ("run_finetunning_xattn|run_contrastive_xattn"
                    "|run_inference_xattn|score_inference")

    def spam_running():
        rc = subprocess.call(["pgrep", "-f", SPAM_PATTERN],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return rc == 0

    def gpu_free_mb(gpu):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free",
                 "--format=csv,noheader,nounits", "-i", str(gpu)], text=True)
            return int(out.strip().splitlines()[0])
        except Exception:
            return 0

    def pick_gpu():
        """First GPU that passes the gates, else None (wait)."""
        if args.spam_guard and spam_running():
            return None
        for g in gpus:
            if gpu_free_mb(g) >= args.min_free_mb:
                return g
        return None

    while todo or running:
        while todo and len(running) < args.parallel:
            gpu = pick_gpu()
            if gpu is None:
                break                  # gated: SPAM active or no memory headroom
            dname, method, out = todo.pop(0)
            if claimed_elsewhere(out):
                print(f"[skip] {dname}/{method} (done or owned by another scheduler)",
                      flush=True)
                continue
            os.makedirs(out, exist_ok=True)
            cmd = job_cmd(method, dname, datasets[dname], out, args, root)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                       SPAM_UNI_WEIGHTS=os.environ.get("SPAM_UNI_WEIGHTS", "/path/to/uni_weights"))
            lg = open(os.path.join(out, "run.log"), "w")
            er = open(os.path.join(out, "run.err"), "w")
            proc = subprocess.Popen(cmd, env=env, stdout=lg, stderr=er)
            with open(os.path.join(out, "started.pid"), "w") as pfh:
                pfh.write(str(proc.pid))
            running.append((proc, dname, method, gpu, time.time(), (lg, er)))
            launched += 1
            print(f"[start {launched}/{total}] gpu{gpu} {dname}/{method} "
                  f"(pid {proc.pid}, {len(running)} in flight)", flush=True)

        time.sleep(20)

        still = []
        for proc, dname, method, gpu, t0, logs in running:
            rc = proc.poll()
            if rc is None:
                still.append((proc, dname, method, gpu, t0, logs))
                continue
            for f in logs:
                f.close()
            done_n += 1
            dt = (time.time() - t0) / 60
            if rc == 0:
                print(f"[done {done_n}/{total}] {dname}/{method} ok ({dt:.1f} min)",
                      flush=True)
            else:
                err = os.path.join(args.out, dname, method, "run.err")
                tail = []
                if os.path.isfile(err):
                    with open(err, errors="replace") as f:
                        tail = [l for l in f.read().splitlines() if l.strip()][-3:]
                print(f"[done {done_n}/{total}] {dname}/{method} FAILED rc={rc} "
                      f"({dt:.1f} min)\n        " + "\n        ".join(tail), flush=True)
        running = still

    print("\nmatrix finished", flush=True)


if __name__ == "__main__":
    main()
