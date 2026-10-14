"""Find where a checkpoint produces NaN/inf on training batches (forward + loss only, no optimizer step).

Runs the training data pipeline (same augmentation as train.py, incl. rot90) and, for each batch, the train-mode
forward pass under fp16 autocast followed by the loss. At the first non-finite result the batch is re-run with a
finiteness check after every layer, in fp16 and in fp32, and the loss is recomputed in fp32 from the fp32
predictions. That separates an fp16 overflow inside a layer from a problem in the loss or the labels.

Usage (Kaggle, one GPU; the weights are those of the failing run):
    python tools/probe_nan.py --weights runs/detect/BladeYOLO_WindSurface/bladeyolo2_v3/weights/last.pt
Output: runs/debug/<name>/{probe_report.json, nan_batch.pt}
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tensors(o):
    if hasattr(o, "is_floating_point"):
        yield o
    elif isinstance(o, (list, tuple)):
        for v in o:
            yield from tensors(v)
    elif isinstance(o, dict):
        for v in o.values():
            yield from tensors(v)


def finite(o):
    return all(bool(t.isfinite().all()) for t in tensors(o) if t.is_floating_point())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="checkpoint (or model YAML) to probe")
    ap.add_argument("--data", default=None)
    ap.add_argument("--batches", type=int, default=100, help="stop after this many batches without a NaN")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--box-loss", choices=("wiou", "ciou"), default="wiou")
    ap.add_argument("--rot90", type=float, default=0.5)
    ap.add_argument("--name", default="probe")
    args = ap.parse_args()

    os.environ["BLADEYOLO_BOX_LOSS"] = args.box_loss
    os.environ["BLADEYOLO_ROT90"] = str(args.rot90)
    sys.path.insert(0, ROOT)
    import torch

    import train
    from models import wiou
    from models.trainer import BladeTrainer

    out = os.path.join(ROOT, "runs", "debug", args.name)
    os.makedirs(out, exist_ok=True)
    cuda = args.device != "cpu" and torch.cuda.is_available()
    tr = BladeTrainer(overrides=dict(
        model=args.weights, data=train.resolve_data(args.data), epochs=1, batch=args.batch, imgsz=args.imgsz,
        device=args.device if cuda else "cpu", workers=2, cache=False, amp=cuda, plots=False,
        project=os.path.join(ROOT, "runs", "debug"), name=args.name, exist_ok=True, **train.RECIPE))
    tr._setup_train()
    model = tr.model
    model.train()
    dev = "cuda" if cuda else "cpu"

    def run(img, amp):
        with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=amp and cuda):
            return model(img)

    def layer_trace(img, amp):
        """Top-level layers (and the first non-finite leaf module) for one forward pass."""
        trace, first_leaf = [], []

        def top(i):
            def hook(mod, inp, o):
                outs = [t for t in tensors(o) if t.is_floating_point()]
                trace.append(dict(layer=i, type=type(mod).__name__, finite=finite(o),
                                  absmax=max((float(t.float().abs().amax()) for t in outs if t.isfinite().all()),
                                             default=None),
                                  in_finite=finite(inp)))
            return hook

        def leaf(name):
            def hook(mod, inp, o):
                if not first_leaf and not finite(o):
                    first_leaf.append(dict(module=name, type=type(mod).__name__, in_finite=finite(inp),
                                           in_absmax=max((float(t.float().abs().amax()) for t in tensors(inp)
                                                          if t.is_floating_point() and t.isfinite().all()),
                                                         default=None)))
            return hook

        hs = [m.register_forward_hook(top(i)) for i, m in enumerate(model.model)]
        hs += [m.register_forward_hook(leaf(n)) for n, m in model.named_modules() if n and not any(m.children())]
        run(img, amp)
        for h in hs:
            h.remove()
        return trace, (first_leaf[0] if first_leaf else None)

    print(f"[probe] {args.weights}: up to {args.batches} training batches, fp16 autocast={cuda}")
    for i, batch in enumerate(tr.train_loader):
        if i >= args.batches:
            print(f"[probe] no NaN/inf in {i} batches")
            break
        batch = tr.preprocess_batch(batch)
        preds = run(batch["img"], amp=True)
        with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=cuda):
            loss, items = model.loss(batch, preds)
        items = {k: float(v) for k, v in items.items()}
        if finite(preds) and bool(loss.isfinite().all()):
            if i % 10 == 0:
                print(f"[probe] batch {i}: ok {items}")
            continue

        trace16, leaf16 = layer_trace(batch["img"], amp=True)
        trace32, leaf32 = layer_trace(batch["img"], amp=False)
        preds32 = run(batch["img"], amp=False)
        with torch.no_grad():
            _, items32 = model.loss(batch, [p.float() for p in preds32] if isinstance(preds32, list) else preds32)
        gt = batch["bboxes"]
        wh = gt[:, 2:4] if gt.numel() else gt
        report = dict(
            weights=args.weights, batch_index=i, im_files=batch.get("im_file"),
            loss_fp16={k: v for k, v in items.items()},
            preds_finite_fp16=finite(preds),
            preds_finite_fp32=finite(preds32),
            loss_fp32={k: float(v) for k, v in items32.items()},
            first_nonfinite_layer_fp16=next((t for t in trace16 if not t["finite"]), None),
            first_nonfinite_leaf_fp16=leaf16,
            first_nonfinite_layer_fp32=next((t for t in trace32 if not t["finite"]), None),
            first_nonfinite_leaf_fp32=leaf32,
            layer_absmax_fp16=[(t["layer"], t["type"], t["absmax"]) for t in trace16],
            labels=dict(n=int(gt.shape[0]), finite=bool(gt.isfinite().all()),
                        min_wh=float(wh.min()) if wh.numel() else None, max_xywh=float(gt.max()) if gt.numel() else None),
            wiou_running_mean=None if wiou._RunningMean.value is None else float(wiou._RunningMean.value),
        )
        report["verdict"] = (
            "fp16 overflow in the forward pass (fp32 forward is finite)" if not report["preds_finite_fp16"]
            and report["preds_finite_fp32"] else
            "forward pass non-finite even in fp32: the weights themselves are broken" if not report["preds_finite_fp32"]
            else "predictions finite: the loss is non-finite (labels or loss function)")
        json.dump(report, open(os.path.join(out, "probe_report.json"), "w"), indent=2, default=str)
        torch.save({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in batch.items()}, os.path.join(out, "nan_batch.pt"))
        print("[probe] NON-FINITE\n" + json.dumps(report, indent=2, default=str))
        return


if __name__ == "__main__":
    main()
