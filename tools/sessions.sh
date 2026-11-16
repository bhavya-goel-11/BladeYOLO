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
#   G29  frozen DINOv3 ViT-B prior on RF-DETR-M ("FP", tools/g29_prior.py; research_log.md G29), each arm paired with an
#        existing RF-DETR run of the same recipe:
#        fp_s0       FP seed 0 on WTBD-group (768, 100 ep) and WTBlade-group (640, 60 ep)        [primary]
#        fp_s1       the same, seed 1                                                            [primary]
#        fp_ws       FP on Wind Surface Defect, official 4:1 protocol (640, 100 ep, 10% carve)   [official]
#        fp_wtbdoff  FP on WTBD-official (768, 100 ep, batch 2 x 8 as its G26 twin)               [official]
#        fp_f25      FP on the four WTBD-group 25% subsets (V1 recipe: 576, 100 ep, RF-DETR's schedule) [secondary]
#        final1 = fp_ws on GPU 0 | fp_s0 on GPU 1      final2 = fp_wtbdoff, fp_f25 on GPU 0 | fp_s1 on GPU 1
#        (two lanes in parallel, one per T4 of a Kaggle session; logs in runs/logs/)
#   Time guard: no run starts, and no training runs on, past BUDGET_H hours (default 11.3) from the session start, so
#   a Kaggle session ends before its 12 h cap with every finished run saved. An unfinished run is simply missing.
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
WS=$BENCH/raw/WindSurface-Defect
WTBDOFF=$BENCH/raw/WTBD-official
export SESSION_T0=${SESSION_T0:-$(date +%s)}
BUDGET_H=${BUDGET_H:-11.3}
MIN_START=${MIN_START:-1800}    # no run starts with less than this many seconds left
PREDICT_RESERVE=${PREDICT_RESERVE:-900}  # training stops this many seconds before the budget ends, for predict

BUDGET_S=$(awk "BEGIN { print int($BUDGET_H * 3600) }")

left() {  # seconds left in the session's time budget
    echo $(( SESSION_T0 + BUDGET_S - $(date +%s) ))
}
ALL="a0s1 a1s1 g28id g28x_wtbd g28x_wtbl yolo12l_wtbl"

prep() {  # prep <folder> <data dir> [copy-paste protocol]: fresh prepared copy in $W (class-agnostic if PRED is set)
    rm -rf "${W:?}/${1:?}"
    python tools/rfdetr_v1.py prepare-official --data "$2" --out "$W/$1" "${PRED[@]}"
    if [ -n "${3:-}" ]; then python tools/g27_paste.py paste --protocol "$3" --dataset "$W/$1"; fi
}

arm() {  # arm <prepared folder in $W, or an absolute dataset path> <run name> <data dir> <resolution> <epochs> [init]
    local last=val sp=()
    if [ ${#SPLITS[@]} -gt 0 ]; then last=${SPLITS[${#SPLITS[@]}-1]}; sp=(--splits "${SPLITS[@]}"); fi
    if [ -f "$R/$2/preds_test.json" ] && [ -f "$R/$2/preds_$last.json" ]; then echo "== $2: done, skipped"; return; fi
    local t; t=$(left)
    if [ "$t" -lt "$MIN_START" ]; then echo "== $2: SKIPPED, $((t / 60)) min left in the ${BUDGET_H} h budget"; return; fi
    echo "== $2 ($(date '+%F %T'), $((t / 60)) min left)"
    local ds="$1"; [ "${ds:0:1}" = / ] || ds="$W/$1"
    mkdir -p "$R/$2"
    { nvidia-smi -L 2>/dev/null || echo "no GPU"; echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"; } > "$R/$2/machine.txt"
    # training stops (SIGINT) 15 min before the budget ends, leaving time for predict; a stopped run is not predicted
    if ! timeout --signal=INT --kill-after=300 $((t - PREDICT_RESERVE)) python tools/rfdetr_v1.py train --dataset "$ds" \
            --out "$R/$2" "${COMMON[@]}" --resolution "$4" --epochs "$5" ${6:+--init "$6"}; then
        # renamed, so a rerun of the queue retrains it instead of predicting from a half-trained checkpoint
        rm -rf "$R/$2_STOPPED"; mv "$R/$2" "$R/$2_STOPPED"
        echo "== $2: TRAINING STOPPED OR FAILED ($(date '+%F %T')), moved to $2_STOPPED; the lane ends here"; return 1
    fi
    python tools/rfdetr_v1.py predict --run "$R/$2" --data "$3" "${PRED[@]}" "${sp[@]}"
}

warm() {  # once per Kaggle session, before the lanes: fail fast, and download shared files once (no race)
    python - <<'PY'
import sys, torch
sys.path.insert(0, "tools")
import g29_prior
from rfdetr import RFDETRMedium
RFDETRMedium()  # downloads rf-detr-medium.pth into the repo root once; both lanes then find it
m = g29_prior.prior_model("b", "cpu")  # DINOv3 source (pip, pinned) + ViT-B weights (Kaggle model beegee11/vitb16)
print(f"warm: RF-DETR-M and DINOv3 ViT-B ok; {torch.cuda.device_count()} GPU(s)")
PY
}

lanes() {  # lanes "<sessions for GPU 0>" "<sessions for GPU 1>": both in parallel, logs in runs/logs/
    warm
    local n; n=$(python -c "import torch; print(torch.cuda.device_count())")
    [ "$n" -ge 2 ] || { echo "lanes need 2 GPUs, found $n (Kaggle: accelerator 'GPU T4 x2')" >&2; exit 2; }
    mkdir -p runs/logs
    local l0="runs/logs/lane0_${1// /+}.log" l1="runs/logs/lane1_${2// /+}.log" rc0=0 rc1=0
    echo "== lanes ($(date '+%F %T')): GPU 0: $1 -> $l0 | GPU 1: $2 -> $l1"
    CUDA_VISIBLE_DEVICES=0 bash tools/sessions.sh $1 > "$l0" 2>&1 & local p0=$!
    CUDA_VISIBLE_DEVICES=1 bash tools/sessions.sh $2 > "$l1" 2>&1 & local p1=$!
    local i=0
    while kill -0 $p0 2>/dev/null || kill -0 $p1 2>/dev/null; do
        sleep 60
        i=$((i + 1))
        if [ $((i % 15)) -eq 0 ]; then
            for l in "$l0" "$l1"; do echo "-- $(date '+%T') $l: $(grep '^== ' "$l" | tail -n 1)"; done
        fi
    done
    wait $p0 || rc0=$?
    wait $p1 || rc1=$?
    for l in "$l0" "$l1"; do echo "---- $l"; grep '^== \|Error\|error:' "$l" | tail -n 25; done
    echo "== lanes finished ($(date '+%F %T')): exit codes $rc0 / $rc1"
}

session() {
    W=$WORK/$1
    COMMON=(--size medium --batch 4 --accum 4 --lr-schedule cosine)  # the G26/G27 recipe on every machine
    PRED=()  # G28 sessions: (--agnostic)
    SPLITS=()  # predict splits (default test val); official protocols: (test valid), the prepared folder's names
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
        fp_s0|fp_s1)  # G29 primary: FP vs A0 (rfdetrm768cos_wtbd_group_full, rfdetrm640cos_wtblade_group_full[_s1])
            local sfx=""; [ "$1" = fp_s1 ] && { COMMON+=(--seed 1); sfx=_s1; }
            COMMON+=(--prior b)
            prep wtbd_group_rf "$WTBD"
            prep wtblade_group_rf "$WTBL"
            arm wtbd_group_rf rfdetrm768cos_fp_wtbd_group_full$sfx "$WTBD" 768 100
            arm wtblade_group_rf rfdetrm640cos_fp_wtblade_group_full$sfx "$WTBL" 640 60
            ;;
        fp_ws)  # G29 official: twin of rfdetrm640cos_windsurface_official (val = test; 10% of train carved, seed 0)
            COMMON+=(--prior b); SPLITS=(test valid)
            rm -rf "${W:?}/ws_off"
            python tools/rfdetr_v1.py prepare-official --data "$WS" --out "$W/ws_off" --carve 0.1
            arm ws_off rfdetrm640cos_fp_windsurface_official "$W/ws_off" 640 100
            ;;
        fp_wtbdoff)  # G29 official: twin of rfdetrm768cos_wtbd_official (which ran batch 2 x accum 8)
            COMMON=(--size medium --batch 2 --accum 8 --lr-schedule cosine --prior b); SPLITS=(test valid)
            prep wtbd_off "$WTBDOFF"
            arm wtbd_off rfdetrm768cos_fp_wtbd_official "$W/wtbd_off" 768 100
            ;;
        fp_f25)  # G29 secondary: FP vs the V1 runs rfdetrm[_randK]_wtbd_group_f25 (576 = Medium's default, RF-DETR's
                 # own schedule, i.e. constant LR over 100 epochs, seed 0)
            COMMON=(--size medium --batch 4 --accum 4 --prior b)
            local k
            for k in "" _rand1 _rand2 _rand3; do
                rm -rf "${W:?}/f25$k"
                python tools/rfdetr_v1.py prepare --data "$WTBD" --subset "tools/splits/wtbd_group_train_f25$k.txt" --out "$W/f25$k"
                arm "f25$k" "rfdetrmfp${k}_wtbd_group_f25" "$WTBD" 576 100
            done
            ;;
        *)
            echo "unknown session: $1 (s2 s3 s4 a0s1 a1s1 g28id g28x_wtbd g28x_wtbl yolo12l_wtbl all final1 final2 fp_*)" >&2
            exit 2
            ;;
    esac
    echo "== session $1 finished ($(date '+%F %T'))"
}

[ $# -gt 0 ] || { echo "usage: bash tools/sessions.sh SESSION [SESSION ...]   (or: all)" >&2; exit 2; }
for s in "$@"; do
    case "$s" in
        all) for a in $ALL; do session "$a"; done ;;
        final1) lanes "fp_ws" "fp_s0" ;;              # G29, Kaggle session 1 (GPU T4 x2)
        final2) lanes "fp_wtbdoff fp_f25" "fp_s1" ;;  # G29, Kaggle session 2
        *) session "$s" ;;
    esac
done
