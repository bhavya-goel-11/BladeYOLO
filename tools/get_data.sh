#!/usr/bin/env bash
# Download the data and pretrained weights this repository needs, into the folders the code reads by default.
# Run from anywhere, in the Python environment of requirements.txt (the script calls `python`, `kaggle` and `hf`).
#
#   bash tools/get_data.sh                    dataset + RF-DETR-M and YOLO12-L COCO checkpoints
#   bash tools/get_data.sh --dino s,b         ... + DINOv3 ViT-S/16 and ViT-B/16 (Hugging Face, gated, see below)
#   bash tools/get_data.sh --verify           ... + check every dataset file against MANIFEST.csv (SHA-256, ~2-5 min)
#   bash tools/get_data.sh --no-data --dino b only the weights
#
# What goes where
#   data/blade-defect-benchmark/          Kaggle dataset beegee11/blade-defect-benchmark (public; 5.9 GB download,
#                                         6.8 GB unpacked, 223,622 files): raw datasets, leak-free splits, pools
#   rf-detr-medium.pth (repo root)        RF-DETR-M COCO checkpoint (Roboflow, Apache-2.0), where tools/rfdetr_v1.py
#   data/weights/rfdetr/rf-detr-medium.pth   looks for it; a copy for tests/ (which run from data/weights/rfdetr)
#   yolo12l.pt (repo root)                YOLO12-L COCO checkpoint (Ultralytics, AGPL-3.0); BladeYOLO-v2 starts from it
#   data/weights/dinov3/vit{s,b,l}16/     DINOv3 from Hugging Face facebook/dinov3-vit{s,b,l}16-pretrain-lvd1689m
#
# Requirements
#   Kaggle: an API token. kaggle.com -> Settings -> API -> "Create New Token" saves kaggle.json; put it at
#           ~/.kaggle/kaggle.json (chmod 600), or export KAGGLE_USERNAME and KAGGLE_KEY.
#   DINOv3: the weights are gated. Accept the DINOv3 License on the model page (e.g.
#           https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m), then `hf auth login` (or export HF_TOKEN).
#   Disk:   about 14 GB free for the dataset (zip + unpacked; the zip is deleted afterwards unless --keep-zip).
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET=beegee11/blade-defect-benchmark
DEST=data/blade-defect-benchmark
DINO=""
GET_DATA=1
KEEP_ZIP=0
VERIFY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dino) DINO=$2; shift 2 ;;
        --no-data) GET_DATA=0; shift ;;
        --keep-zip) KEEP_ZIP=1; shift ;;
        --verify) VERIFY=1; shift ;;
        -h|--help) sed -n 2,23p "$0"; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
done

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing command '$1': $2" >&2; exit 1; }; }

if [ "$GET_DATA" = 1 ]; then
    if [ -f "$DEST/MANIFEST.csv" ]; then
        echo "== dataset: $DEST already present, skipped"
    else
        need kaggle "pip install kaggle (it is in requirements.txt) and add your API token, see --help"
        need unzip "e.g. sudo apt install unzip"
        free_kb=$(df -Pk data 2>/dev/null | awk 'NR==2 {print $4}' || df -Pk . | awk 'NR==2 {print $4}')
        mkdir -p data
        if [ "${free_kb:-0}" -lt $((14 * 1024 * 1024)) ]; then
            echo "warning: only $((free_kb / 1024 / 1024)) GB free; about 14 GB are needed while unpacking" >&2
        fi
        echo "== dataset: downloading $DATASET (5.9 GB) to data/ ($(date '+%T'))"
        kaggle datasets download "$DATASET" -p data
        echo "== dataset: unpacking to $DEST ($(date '+%T'))"
        mkdir -p "$DEST"
        unzip -q -o data/blade-defect-benchmark.zip -d "$DEST"
        [ "$KEEP_ZIP" = 1 ] || rm -f data/blade-defect-benchmark.zip
        n=$(($(wc -l < "$DEST/MANIFEST.csv") - 1))
        echo "== dataset: done, MANIFEST lists $n files ($(date '+%T'))"
    fi
    if [ "$VERIFY" = 1 ]; then
        echo "== dataset: verifying SHA-256 of every file in MANIFEST.csv"
        python - "$DEST" <<'PY'
import csv, hashlib, os, sys
root = sys.argv[1]
bad = missing = 0
rows = list(csv.DictReader(open(os.path.join(root, "MANIFEST.csv"))))
for i, r in enumerate(rows):
    p = os.path.join(root, r["path"])
    if not os.path.exists(p):
        missing += 1
        continue
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    if h.hexdigest() != r["sha256"]:
        bad += 1
        print("mismatch:", r["path"])
    if (i + 1) % 50000 == 0:
        print(f"  {i + 1}/{len(rows)}", flush=True)
print(f"verified {len(rows)} files: {missing} missing, {bad} mismatched")
sys.exit(1 if bad or missing else 0)
PY
    fi
fi

echo "== weights: RF-DETR-M COCO checkpoint"
if [ -f rf-detr-medium.pth ]; then echo "   rf-detr-medium.pth present"
elif [ -f data/weights/rfdetr/rf-detr-medium.pth ]; then cp data/weights/rfdetr/rf-detr-medium.pth .
else
    python -c "from rfdetr import RFDETRMedium; RFDETRMedium()" >/dev/null
fi
mkdir -p data/weights/rfdetr
[ -f data/weights/rfdetr/rf-detr-medium.pth ] || cp rf-detr-medium.pth data/weights/rfdetr/

echo "== weights: YOLO12-L COCO checkpoint"
if [ -f yolo12l.pt ]; then echo "   yolo12l.pt present"
else
    python -c "from ultralytics import YOLO; YOLO('yolo12l.pt')" >/dev/null
fi

if [ -n "$DINO" ]; then
    need hf "pip install -U huggingface_hub (installed with transformers), then: hf auth login"
    for s in ${DINO//,/ }; do
        case "$s" in s|b|l) ;; *) echo "--dino takes s, b and/or l, got '$s'" >&2; exit 2 ;; esac
        out=data/weights/dinov3/vit${s}16
        if [ -f "$out/model.safetensors" ]; then echo "== weights: DINOv3 ViT-${s^^}/16 present"; continue; fi
        echo "== weights: DINOv3 ViT-${s^^}/16 from facebook/dinov3-vit${s}16-pretrain-lvd1689m"
        hf download "facebook/dinov3-vit${s}16-pretrain-lvd1689m" --local-dir "$out" || {
            echo "DINOv3 download failed: accept the licence on the model page and run 'hf auth login' (see --help)" >&2
            exit 1; }
    done
fi
echo "== all done"
