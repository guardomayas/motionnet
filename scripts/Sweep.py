"""Run every point of an experiment grid, resumably.

    python scripts/sweep.py s02_constraints            # run / resume the sweep
    python scripts/sweep.py s02_constraints --dry-run  # list runs, train nothing

An experiment is configs/exp/<name>.yaml:

    name: s02_constraints
    data: eye31_150fps          # configs/stimulus/<data>.yaml
    model: base                 # configs/model/<model>.yaml
    fixed:                      # optional: overrides applied to every run
      loss.lambda_sp: 0.01
    grid:                       # full factorial over these
      loss.gamma_var: [0.0, 0.5, 1.0]
      optim.seed: [0, 1, 2]

Outputs go to <ckpt>/<name>/:
    <tag>.pt, <tag>_cfg.json   checkpoint + resolved config (from fit)
    <tag>.err                  traceback, if that run failed
    summary.csv                one row per finished run (rebuilt every time)
    manifest.json              git commit, grid, start time

A run whose <tag>.pt exists is skipped, so relaunching resumes the sweep.
Delete a .pt (or the whole folder) to force a rerun.
"""

import argparse
import csv
import itertools
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import torch

from motionnet.dataset import get_data
from motionnet.paths import REPO, compose, setup
from motionnet.train import fit

# final-epoch validation metrics copied into summary.csv
METRICS = ("nmse", "r2_x", "r2_y", "mse", "svar", "zE", "rate1", "rate2")


def in_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def git_state():
    """Commit hash, with '-dirty' if the working tree has uncommitted changes."""
    try:
        rev = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                      text=True).strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD"]).returncode
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def tag_for(over):
    """{'loss.gamma_var': 0.5, 'optim.seed': 1} -> 'gamma_var-0.5_seed-1'"""
    return "_".join(f"{k.split('.')[-1]}-{v}" for k, v in over.items())


def summary_row(ckpt_path, over):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    final = ck["history"]["val"][-1]
    row = {"tag": ck["tag"], **over}
    row.update({m: final.get(m, float("nan")) for m in METRICS})
    row["wall_s"] = round(ck.get("wall_s", float("nan")), 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", help="name of configs/exp/<exp>.yaml")
    ap.add_argument("--dry-run", action="store_true", help="list runs without training")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    P = setup(in_colab())
    _, grid = compose(exp_name=args.exp, corpus=P.corpus)
    if not grid:
        raise SystemExit(f"configs/exp/{args.exp}.yaml has no grid")
    points = [dict(zip(grid, vals)) for vals in itertools.product(*grid.values())]

    out = Path(P.ckpt) / args.exp
    out.mkdir(parents=True, exist_ok=True)
    commit = git_state()
    print(f"{args.exp}: {len(points)} runs -> {out}   (code {commit})")
    if commit.endswith("-dirty"):
        print("WARNING: uncommitted changes; these runs can't be reproduced from a commit")

    if args.dry_run:
        for over in points:
            done = (out / f"{tag_for(over)}.pt").exists()
            print(f"  {'done' if done else 'todo'}  {tag_for(over)}")
        return

    (out / "manifest.json").write_text(json.dumps(
        dict(exp=args.exp, commit=commit, grid=grid, n_runs=len(points),
             started=time.strftime("%Y-%m-%d %H:%M:%S"), device=args.device),
        indent=1, default=str))

    data, data_key = None, None
    for i, over in enumerate(points, 1):
        tag = tag_for(over)
        if (out / f"{tag}.pt").exists():
            print(f"[{i}/{len(points)}] {tag}: done, skipping")
            continue

        cfg, _ = compose(exp_name=args.exp, corpus=P.corpus, **over)
        cfg["meta"].update(commit=commit, tag=tag, grid_point=over)

        # Load stimuli once; reload only if a grid axis changes the stimulus.
        key = json.dumps(cfg["stimulus"], sort_keys=True, default=str)
        if key != data_key:
            data = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            data, data_key = get_data(cfg["stimulus"], P.movies, args.device), key

        print(f"[{i}/{len(points)}] {tag}")
        try:
            fit(cfg, data, args.device, out, tag=tag, progress=False, log_every=0)
            (out / f"{tag}.err").unlink(missing_ok=True)
        except Exception:
            (out / f"{tag}.err").write_text(traceback.format_exc())
            print(f"  FAILED (see {tag}.err), continuing")
            if "CUDA" in traceback.format_exc():
                raise SystemExit("CUDA error: the GPU context is unusable; "
                                 "restart the runtime and relaunch to resume")

    # Rebuild the summary from every finished checkpoint.
    rows = [summary_row(out / f"{tag_for(o)}.pt", o)
            for o in points if (out / f"{tag_for(o)}.pt").exists()]
    if rows:
        with open(out / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    failed = [tag_for(o) for o in points if (out / f"{tag_for(o)}.err").exists()]
    print(f"finished {len(rows)}/{len(points)}; summary -> {out / 'summary.csv'}")
    if failed:
        print("failed:", ", ".join(failed))


if __name__ == "__main__":
    main()