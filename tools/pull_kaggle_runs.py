"""Pull the runs/ directory from the latest version of the Kaggle notebook and merge it into ./runs.

Only files under a runs/ directory are downloaded (the notebook output also holds the repo clone and the
dataset copy, which are not needed). Files are staged in ./.kaggle_pull and merged file by file:

  new file              copied
  identical file        skipped
  different file        kept as is locally and reported (--overwrite replaces it)

The notebook's execution log is kept as runs/kaggle_logs/<kernel>_<UTC time>.log.

  .venv/bin/python tools/pull_kaggle_runs.py                       # beegee11/bladeyolo
  .venv/bin/python tools/pull_kaggle_runs.py --dry-run             # show what would change
  .venv/bin/python tools/pull_kaggle_runs.py --kernel owner/name --overwrite
"""

import argparse
import datetime
import filecmp
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
STAGE = os.path.join(ROOT, ".kaggle_pull")


def kaggle_cli():
    exe = os.path.join(os.path.dirname(sys.executable), "kaggle")  # the project venv's CLI
    return exe if os.path.exists(exe) else (shutil.which("kaggle") or sys.exit("kaggle CLI not found: .venv/bin/pip install kaggle"))


def run(*args):
    r = subprocess.run([kaggle_cli(), *args], capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"kaggle {' '.join(args)} failed:\n{r.stdout}{r.stderr}")
    return r.stdout


def runs_roots(stage):
    """Top-level runs/ directories in the download (nested runs/ folders belong to their parent)."""
    roots = []
    for d, sub, _ in os.walk(stage):
        if os.path.basename(d) == "runs":
            roots.append(d)
            sub[:] = []
    return roots


def merge(src, dst, overwrite, dry):
    counts = {"new": 0, "same": 0, "replaced": 0, "conflict": 0}
    conflicts = []
    for d, _, files in os.walk(src):
        for f in files:
            s = os.path.join(d, f)
            t = os.path.join(dst, os.path.relpath(s, src))
            if not os.path.exists(t):
                kind = "new"
            elif filecmp.cmp(s, t, shallow=False):
                counts["same"] += 1
                continue
            elif overwrite:
                kind = "replaced"
            else:
                counts["conflict"] += 1
                conflicts.append(os.path.relpath(t, ROOT))
                continue
            counts[kind] += 1
            print(f"  {kind:8s} {os.path.relpath(t, ROOT)}")
            if not dry:
                os.makedirs(os.path.dirname(t), exist_ok=True)
                shutil.copy2(s, t)
    return counts, conflicts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernel", default="beegee11/bladeyolo", help="owner/kernel-name (optionally /version)")
    ap.add_argument("--overwrite", action="store_true", help="replace local files that differ from the download")
    ap.add_argument("--dry-run", action="store_true", help="download and report, change nothing in runs/")
    ap.add_argument("--keep-download", action="store_true", help="leave the staged download in .kaggle_pull/")
    args = ap.parse_args()

    status = run("kernels", "status", args.kernel).strip()
    print(status)
    if "COMPLETE" not in status:
        print("warning: the latest version has not completed; its output may be missing or from an earlier version")

    shutil.rmtree(STAGE, ignore_errors=True)
    os.makedirs(STAGE)
    print(f"downloading runs/ from {args.kernel} ...")
    run("kernels", "output", args.kernel, "-p", STAGE, "-o", "-q", "--file-pattern", r"(^|/)runs/")

    roots = runs_roots(STAGE)
    if not roots:
        sys.exit("no runs/ directory in the notebook output")
    total, conflicts = {}, []
    for r in roots:
        print(f"merging {os.path.relpath(r, STAGE)} -> runs/")
        c, cf = merge(r, RUNS, args.overwrite, args.dry_run)
        total = {k: total.get(k, 0) + v for k, v in c.items()}
        conflicts += cf

    logs = [f for f in os.listdir(STAGE) if f.endswith(".log")]
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    for f in logs:
        t = os.path.join(RUNS, "kaggle_logs", f"{args.kernel.split('/')[1]}_{stamp}.log")
        print(f"  log      {os.path.relpath(t, ROOT)}")
        if not args.dry_run:
            os.makedirs(os.path.dirname(t), exist_ok=True)
            shutil.copy2(os.path.join(STAGE, f), t)

    if not args.keep_download:
        shutil.rmtree(STAGE, ignore_errors=True)
    print(("dry run: " if args.dry_run else "") + ", ".join(f"{v} {k}" for k, v in total.items()))
    if conflicts:
        print(f"{len(conflicts)} local files differ from the download and were kept (rerun with --overwrite to replace):")
        for c in conflicts:
            print("  " + c)


if __name__ == "__main__":
    main()
