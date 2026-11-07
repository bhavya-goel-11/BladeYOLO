#!/usr/bin/env bash
# Training sessions of the BladeYOLO study (research_log.md: G27, G28 and baselines). Run from the repo root, in the
# environment that has requirements.txt installed (the script calls `python` and nothing else):
#   bash tools/sessions.sh all                  every remaining run, one after another
#   bash tools/sessions.sh a1s1 g28id           any sessions, in order
#   Kaggle (one notebook line): !bash tools/sessions.sh a1s1
# Sessions:
#   G27  s2    A0, A1 on WTBD-group; A0 on WTBlade-group      s3  A2, A3 on WTBD-group; A1 on WTBlade-group
#        s4    A2, A3 on WTBlade-group                        a0s1  A0 seed 1 on both datasets (A-2)
#        a1s1  R = A1 (A-1, 8 Nov) seed 1 on both datasets (A-2)
#   G28  g28id  ID-WTBD and ID-WTBlade    g28x_wtbd  X-WTBD    g28x_wtbl  X-WTBlade   (class-agnostic, 640 px)
#   baseline  yolo12l_wtbl  YOLO12-L on WTBlade-group (train.py; the architecture comparison)
#   all   a0s1 a1s1 g28id g28x_wtbd g28x_wtbl yolo12l_wtbl
# Data: the dataset blade-defect-benchmark, at /kaggle/input/datasets/beegee11/blade-defect-benchmark on Kaggle and at
#   data/blade-defect-benchmark otherwise (BENCH=... overrides). Prepared copies go to /kaggle/working/<session>/ or
#   runs/work/<session>/ (WORK=... overrides).
# A failing command stops the script. A finished run (RF-DETR: preds_test.json + preds_val.json; YOLO:
#   test/metrics.json) is skipped, so running the same line again resumes an interrupted queue.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -d /kaggle/input ]; then
    BENCH=${BENCH:-/kaggle/input/datasets/beegee11/blade-defect-benchmark}
    WORK=${WORK:-/kaggle/working}
else
    BENCH=${BENCH:-$PWD/data/blade-defect-benchmark}
    WORK=${WORK:-$PWD/runs/work}
fi
WTBD=$BENCH/leakfree/WTBD-group
WTBL=$BENCH/leakfree/WTBlade-group
D1=$BENCH/checkpoints/g27-s1/rfdetrm_g27ssl.pth
D2=$BENCH/checkpoints/g27-s1/last_ema.pth
XWTBD=$BENCH/pools/g28-x-wtbd
XWTBL=$BENCH/pools/g28-x-wtblade
R=${RUNS:-runs/detect/BladeYOLO_WindSurface}
ALL="a0s1 a1s1 g28id g28x_wtbd g28x_wtbl yolo12l_wtbl"

prep() {  # prep <folder> <data dir> [copy-paste protocol]: fresh prepared copy in $W (class-agnostic if PRED is set)
    rm -rf "${W:?}/${1:?}"
    python tools/rfdetr_v1.py prepare-official --data "$2" --out "$W/$1" "${PRED[@]}"
    if [ -n "${3:-}" ]; then python tools/g27_paste.py paste --protocol "$3" --dataset "$W/$1"; fi
}

arm() {  # arm <prepared folder in $W, or an absolute dataset path> <run name> <data dir> <resolution> <epochs> [init]
    if [ -f "$R/$2/preds_test.json" ] && [ -f "$R/$2/preds_val.json" ]; then echo "== $2: done, skipped"; return; fi
    echo "== $2 ($(date '+%F %T'))"
    local ds="$1"; [ "${ds:0:1}" = / ] || ds="$W/$1"
    python tools/rfdetr_v1.py train --dataset "$ds" --out "$R/$2" "${COMMON[@]}" --resolution "$4" --epochs "$5" \
        ${6:+--init "$6"}
    python tools/rfdetr_v1.py predict --run "$R/$2" --data "$3" "${PRED[@]}"
}

session() {
    W=$WORK/$1
    COMMON=(--size medium --batch 4 --accum 4 --lr-schedule cosine)  # the G26/G27 recipe on every machine
    PRED=()  # G28 sessions: (--agnostic)
    echo "== session $1 ($(date '+%F %T')): data $BENCH"
    case "$1" in
        s2)
            prep wtbd_group_rf "$WTBD"
            prep wtbd_group_cp "$WTBD" wtbd_group
            prep wtblade_group_rf "$WTBL"
            arm wtbd_group_rf rfdetrm768cos_wtbd_group_full "$WTBD" 768 100            # A0
            arm wtbd_group_cp rfdetrm768cos_cp_wtbd_group_full "$WTBD" 768 100         # A1
            arm wtblade_group_rf rfdetrm640cos_wtblade_group_full "$WTBL" 640 60       # A0
            ;;
        s3)
            prep wtbd_group_cp "$WTBD" wtbd_group
            prep wtblade_group_cp "$WTBL" wtblade_group
            arm wtbd_group_cp rfdetrm768cos_cp_d1_wtbd_group_full "$WTBD" 768 100 "$D1"  # A2
            arm wtbd_group_cp rfdetrm768cos_cp_d2_wtbd_group_full "$WTBD" 768 100 "$D2"  # A3
            arm wtblade_group_cp rfdetrm640cos_cp_wtblade_group_full "$WTBL" 640 60      # A1
            ;;
        s4)
            prep wtblade_group_cp "$WTBL" wtblade_group
            arm wtblade_group_cp rfdetrm640cos_cp_d1_wtblade_group_full "$WTBL" 640 60 "$D1"  # A2
            arm wtblade_group_cp rfdetrm640cos_cp_d2_wtblade_group_full "$WTBL" 640 60 "$D2"  # A3
            ;;
        a0s1)  # A-2's seed-1 A0 runs: A0 is in A-2 whatever R is, and R is picked on seed-0 val only
            COMMON+=(--seed 1)
            prep wtbd_group_rf "$WTBD"
            prep wtblade_group_rf "$WTBL"
            arm wtbd_group_rf rfdetrm768cos_wtbd_group_full_s1 "$WTBD" 768 100            # A0, seed 1
            arm wtblade_group_rf rfdetrm640cos_wtblade_group_full_s1 "$WTBL" 640 60       # A0, seed 1
            ;;
        a1s1)  # A-2's seed-1 R runs; A-1 picked R = A1 (copy-paste) on seed-0 val
            COMMON+=(--seed 1)
            prep wtbd_group_cp "$WTBD" wtbd_group
            prep wtblade_group_cp "$WTBL" wtblade_group
            arm wtbd_group_cp rfdetrm768cos_cp_wtbd_group_full_s1 "$WTBD" 768 100          # A1, seed 1
            arm wtblade_group_cp rfdetrm640cos_cp_wtblade_group_full_s1 "$WTBL" 640 60     # A1, seed 1
            ;;
        g28x_wtbd)  # X-WTBD: every other source, twin-free against all of WTBD; scored on WTBD-group test
            PRED=(--agnostic)
            arm "$XWTBD" rfdetrm640ag_x_wtbd_group "$WTBD" 640 12
            ;;
        g28x_wtbl)  # X-WTBlade: every other source, twin-free against all of WTBlade-Defect; scored on WTBlade-group test
            PRED=(--agnostic)
            arm "$XWTBL" rfdetrm640ag_x_wtblade_group "$WTBL" 640 12
            ;;
        g28id)  # ID-T: the target's own group train, class-agnostic (G27 A0's epochs)
            PRED=(--agnostic)
            prep wtbd_group_ag "$WTBD"
            prep wtblade_group_ag "$WTBL"
            arm wtbd_group_ag rfdetrm640ag_id_wtbd_group "$WTBD" 640 100
            arm wtblade_group_ag rfdetrm640ag_id_wtblade_group "$WTBL" 640 60
            ;;
        yolo12l_wtbl)  # the architecture-comparison baseline missing on WTBlade-group (the §12 recipe, 50 epochs)
            if [ -f "$R/yolo12l_wtblade_group/test/metrics.json" ]; then echo "== yolo12l_wtblade_group: done, skipped"
            else
                echo "== yolo12l_wtblade_group ($(date '+%F %T'))"
                python train.py --data "$WTBL/data.yaml" --model yolo12l.pt --box-loss ciou --name yolo12l_wtblade_group --epochs 50
            fi
            ;;
        *)
            echo "unknown session: $1 (s2 s3 s4 a0s1 a1s1 g28id g28x_wtbd g28x_wtbl yolo12l_wtbl all)" >&2
            exit 2
            ;;
    esac
    echo "== session $1 finished ($(date '+%F %T'))"
}

[ $# -gt 0 ] || { echo "usage: bash tools/sessions.sh SESSION [SESSION ...]   (or: all)" >&2; exit 2; }
for s in "$@"; do
    if [ "$s" = all ]; then for a in $ALL; do session "$a"; done; else session "$s"; fi
done
