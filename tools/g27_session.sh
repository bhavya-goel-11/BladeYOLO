#!/usr/bin/env bash
# G27 phase-A Kaggle sessions (research_log.md, "G27 Kaggle commands"), one line per notebook, from the repo root:
#   !bash tools/g27_session.sh s2   # A0, A1 on WTBD-group; A0 on WTBlade-group
#   !bash tools/g27_session.sh s3   # A2, A3 on WTBD-group; A1 on WTBlade-group (needs the S1 checkpoints)
#   !bash tools/g27_session.sh s4   # A2, A3 on WTBlade-group (needs the S1 checkpoints)
#   !bash tools/g27_session.sh a0s1 # A0 seed 1 on both datasets (A-2)
# G28 (leave-one-dataset-out, class-agnostic, 640 px):
#   !bash tools/g27_session.sh g28x_wtbd   # X-WTBD (input beegee11/g28-x-wtbd)
#   !bash tools/g27_session.sh g28x_wtbl   # X-WTBlade (input beegee11/g28-x-wtblade)
#   !bash tools/g27_session.sh g28id       # ID-WTBD and ID-WTBlade
# Inputs: wind-surface-defect and blade-group-splits; s3/s4 also beegee11/g27-s1 (or set D1=... D2=... before the line).
# Every variable is set here, so nothing depends on notebook cell magics. Any failing step stops the session.
# A run whose preds_test.json and preds_val.json already exist is skipped, so rerunning a timed-out session resumes it.
set -euo pipefail
cd "$(dirname "$0")/.."

WTBD=${WTBD:-/kaggle/input/datasets/beegee11/wind-surface-defect/WTBD-group}
WTBL=${WTBL:-/kaggle/input/datasets/beegee11/blade-group-splits}
D1=${D1:-/kaggle/input/datasets/beegee11/g27-s1/rfdetrm_g27ssl.pth}
D2=${D2:-/kaggle/input/datasets/beegee11/g27-s1/last_ema.pth}
XWTBD=${XWTBD:-/kaggle/input/datasets/beegee11/g28-x-wtbd}
XWTBL=${XWTBL:-/kaggle/input/datasets/beegee11/g28-x-wtblade}
W=${WORK:-/kaggle/working}
R=${G27_RUNS:-runs/detect/BladeYOLO_WindSurface}  # overridden only by tests
COMMON=(--size medium --batch 4 --accum 4 --lr-schedule cosine)
PRED=()  # G28 sessions: (--agnostic)

need() { for f in "$@"; do [ -e "$f" ] || { echo "missing input: $f" >&2; exit 1; }; done; }

prep() {  # prep <folder> <data dir> [copy-paste protocol]: fresh prepared copy in $W (class-agnostic if PRED is set)
    rm -rf "${W:?}/${1:?}"
    python tools/rfdetr_v1.py prepare-official --data "$2" --out "$W/$1" "${PRED[@]}"
    if [ -n "${3:-}" ]; then python tools/g27_paste.py paste --protocol "$3" --dataset "$W/$1"; fi
}

arm() {  # arm <prepared folder in $W, or an absolute dataset path> <run name> <data dir> <resolution> <epochs> [init]
    if [ -f "$R/$2/preds_test.json" ] && [ -f "$R/$2/preds_val.json" ]; then echo "== $2: done, skipped"; return; fi
    echo "== $2"
    local ds="$1"; [ "${ds:0:1}" = / ] || ds="$W/$1"
    python tools/rfdetr_v1.py train --dataset "$ds" --out "$R/$2" "${COMMON[@]}" --resolution "$4" --epochs "$5" \
        ${6:+--init "$6"}
    python tools/rfdetr_v1.py predict --run "$R/$2" --data "$3" "${PRED[@]}"
}

case "${1:-}" in
    s2)
        need "$WTBD/data.yaml" "$WTBL/data.yaml"
        prep wtbd_group_rf "$WTBD"
        prep wtbd_group_cp "$WTBD" wtbd_group
        prep wtblade_group_rf "$WTBL"
        arm wtbd_group_rf rfdetrm768cos_wtbd_group_full "$WTBD" 768 100            # A0
        arm wtbd_group_cp rfdetrm768cos_cp_wtbd_group_full "$WTBD" 768 100         # A1
        arm wtblade_group_rf rfdetrm640cos_wtblade_group_full "$WTBL" 640 60       # A0
        ;;
    s3)
        need "$WTBD/data.yaml" "$WTBL/data.yaml" "$D1" "$D2"
        prep wtbd_group_cp "$WTBD" wtbd_group
        prep wtblade_group_cp "$WTBL" wtblade_group
        arm wtbd_group_cp rfdetrm768cos_cp_d1_wtbd_group_full "$WTBD" 768 100 "$D1"  # A2
        arm wtbd_group_cp rfdetrm768cos_cp_d2_wtbd_group_full "$WTBD" 768 100 "$D2"  # A3
        arm wtblade_group_cp rfdetrm640cos_cp_wtblade_group_full "$WTBL" 640 60      # A1
        ;;
    s4)
        need "$WTBL/data.yaml" "$D1" "$D2"
        prep wtblade_group_cp "$WTBL" wtblade_group
        arm wtblade_group_cp rfdetrm640cos_cp_d1_wtblade_group_full "$WTBL" 640 60 "$D1"  # A2
        arm wtblade_group_cp rfdetrm640cos_cp_d2_wtblade_group_full "$WTBL" 640 60 "$D2"  # A3
        ;;
    a0s1)  # A-2's seed-1 A0 runs: A0 is in A-2 whatever R is, and R is picked on seed-0 val only
        need "$WTBD/data.yaml" "$WTBL/data.yaml"
        COMMON+=(--seed 1)
        prep wtbd_group_rf "$WTBD"
        prep wtblade_group_rf "$WTBL"
        arm wtbd_group_rf rfdetrm768cos_wtbd_group_full_s1 "$WTBD" 768 100            # A0, seed 1
        arm wtblade_group_rf rfdetrm640cos_wtblade_group_full_s1 "$WTBL" 640 60       # A0, seed 1
        ;;
    g28x_wtbd)  # X-WTBD: every other source, twin-free against all of WTBD; scored on WTBD-group test
        need "$XWTBD/data.yaml" "$WTBD/data.yaml"
        PRED=(--agnostic)
        arm "$XWTBD" rfdetrm640ag_x_wtbd_group "$WTBD" 640 12
        ;;
    g28x_wtbl)  # X-WTBlade: every other source, twin-free against all of WTBlade-Defect; scored on WTBlade-group test
        need "$XWTBL/data.yaml" "$WTBL/data.yaml"
        PRED=(--agnostic)
        arm "$XWTBL" rfdetrm640ag_x_wtblade_group "$WTBL" 640 12
        ;;
    g28id)  # ID-T: the target's own group train, class-agnostic (G27 A0's epochs)
        need "$WTBD/data.yaml" "$WTBL/data.yaml"
        PRED=(--agnostic)
        prep wtbd_group_ag "$WTBD"
        prep wtblade_group_ag "$WTBL"
        arm wtbd_group_ag rfdetrm640ag_id_wtbd_group "$WTBD" 640 100
        arm wtblade_group_ag rfdetrm640ag_id_wtblade_group "$WTBL" 640 60
        ;;
    *)
        echo "usage: bash tools/g27_session.sh s2|s3|s4|a0s1|g28x_wtbd|g28x_wtbl|g28id" >&2
        exit 2
        ;;
esac
echo "== session $1 finished"
