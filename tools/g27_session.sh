#!/usr/bin/env bash
# G27/G28 training sessions (research_log.md), on Kaggle or on a local GPU machine. From the repo root:
#   Kaggle (one notebook line):  !bash tools/g27_session.sh a1s1
#   local (several sessions run one after another; one GPU is enough):
#       nohup bash tools/g27_session.sh a1s1 g28id g28x_wtbd g28x_wtbl > runs/logs/queue.log 2>&1 &
# Sessions:
#   s2    A0, A1 on WTBD-group; A0 on WTBlade-group           s3  A2, A3 on WTBD-group; A1 on WTBlade-group
#   s4    A2, A3 on WTBlade-group                             a0s1  A0 seed 1 on both datasets (G27 A-2)
#   a1s1  R = A1 (G27 A-1, 8 Nov) seed 1 on both datasets (G27 A-2)
#   g28x_wtbd  X-WTBD    g28x_wtbl  X-WTBlade    g28id  ID-WTBD and ID-WTBlade   (G28, class-agnostic, 640 px)
# Data: the dataset blade-defect-benchmark (raw datasets, leak-free splits, pools, S1 checkpoints). On Kaggle it is
#   attached as an input; locally it is expected at data/blade-defect-benchmark (gitignored). BENCH=... overrides
#   either, and every derived path (WTBD, WTBL, D1, D2, XWTBD, XWTBL) can be overridden too.
# Python: `python` on Kaggle; locally .venv-rfdetr/bin/python if it exists, else `python`; PYTHON=... overrides.
# COCO weights: RF-DETR downloads rf-detr-medium.pth into RF_HOME (default ~/.roboflow/models). Locally, without
#   internet, put that file in data/weights/rfdetr/; the script then sets RF_HOME there.
# Work folders (prepared dataset copies) are per session: /kaggle/working/<session>/ or runs/work/<session>/.
# Every variable is set here, so nothing depends on notebook cell magics. Any failing step stops the script.
# A run whose preds_test.json and preds_val.json already exist is skipped, so rerunning an interrupted session resumes.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -d /kaggle/input ]; then
    ON=kaggle
    BENCH=${BENCH:-/kaggle/input/datasets/beegee11/blade-defect-benchmark}
    WORK=${WORK:-/kaggle/working}
    PYTHON=${PYTHON:-python}
else
    ON=local
    BENCH=${BENCH:-$PWD/data/blade-defect-benchmark}
    WORK=${WORK:-$PWD/runs/work}
    if [ -z "${PYTHON:-}" ]; then
        if [ -x .venv-rfdetr/bin/python ]; then PYTHON=$PWD/.venv-rfdetr/bin/python; else PYTHON=python; fi
    fi
    if [ -z "${RF_HOME:-}" ] && [ -f data/weights/rfdetr/rf-detr-medium.pth ]; then export RF_HOME=$PWD/data/weights/rfdetr; fi
fi
WTBD=${WTBD:-$BENCH/leakfree/WTBD-group}
WTBL=${WTBL:-$BENCH/leakfree/WTBlade-group}
D1=${D1:-$BENCH/checkpoints/g27-s1/rfdetrm_g27ssl.pth}
D2=${D2:-$BENCH/checkpoints/g27-s1/last_ema.pth}
XWTBD=${XWTBD:-$BENCH/pools/g28-x-wtbd}
XWTBL=${XWTBL:-$BENCH/pools/g28-x-wtblade}
R=${G27_RUNS:-runs/detect/BladeYOLO_WindSurface}  # overridden only by tests
SESSIONS="s2 s3 s4 a0s1 a1s1 g28x_wtbd g28x_wtbl g28id"

need() { for f in "$@"; do [ -e "$f" ] || { echo "missing input: $f" >&2; exit 1; }; done; }

prep() {  # prep <folder> <data dir> [copy-paste protocol]: fresh prepared copy in $W (class-agnostic if PRED is set)
    rm -rf "${W:?}/${1:?}"
    "$PYTHON" tools/rfdetr_v1.py prepare-official --data "$2" --out "$W/$1" "${PRED[@]}"
    if [ -n "${3:-}" ]; then "$PYTHON" tools/g27_paste.py paste --protocol "$3" --dataset "$W/$1"; fi
}

arm() {  # arm <prepared folder in $W, or an absolute dataset path> <run name> <data dir> <resolution> <epochs> [init]
    if [ -f "$R/$2/preds_test.json" ] && [ -f "$R/$2/preds_val.json" ]; then echo "== $2: done, skipped"; return; fi
    echo "== $2 ($(date '+%F %T'))"
    local ds="$1"; [ "${ds:0:1}" = / ] || ds="$W/$1"
    "$PYTHON" tools/rfdetr_v1.py train --dataset "$ds" --out "$R/$2" "${COMMON[@]}" --resolution "$4" --epochs "$5" \
        ${6:+--init "$6"}
    "$PYTHON" tools/rfdetr_v1.py predict --run "$R/$2" --data "$3" "${PRED[@]}"
}

session() {
    W=$WORK/$1
    COMMON=(--size medium --batch 4 --accum 4 --lr-schedule cosine)  # the G26/G27 recipe on every machine
    PRED=()  # G28 sessions: (--agnostic)
    echo "== session $1 on $ON: data $BENCH, work $W, python $PYTHON"
    case "$1" in
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
        a1s1)  # A-2's seed-1 R runs; A-1 picked R = A1 (copy-paste) on seed-0 val
            need "$WTBD/data.yaml" "$WTBL/data.yaml"
            COMMON+=(--seed 1)
            prep wtbd_group_cp "$WTBD" wtbd_group
            prep wtblade_group_cp "$WTBL" wtblade_group
            arm wtbd_group_cp rfdetrm768cos_cp_wtbd_group_full_s1 "$WTBD" 768 100          # A1, seed 1
            arm wtblade_group_cp rfdetrm640cos_cp_wtblade_group_full_s1 "$WTBL" 640 60     # A1, seed 1
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
    esac
    echo "== session $1 finished ($(date '+%F %T'))"
}

[ $# -gt 0 ] || { echo "usage: bash tools/g27_session.sh SESSION [SESSION ...]   ($SESSIONS)" >&2; exit 2; }
for s in "$@"; do  # check every name before any work, so a typo does not surface hours into a queue
    case " $SESSIONS " in *" $s "*) ;; *) echo "unknown session: $s   ($SESSIONS)" >&2; exit 2;; esac
done
for s in "$@"; do session "$s"; done
