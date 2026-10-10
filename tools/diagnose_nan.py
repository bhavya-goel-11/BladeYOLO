"""Find where BladeYOLO training produces inf/NaN under fp16 mixed precision (run on the GPU).

Single-GPU training with train.py's RECIPE and the 300-epoch schedule, stopped after --stop-epoch epochs.
  * every epoch: max |activation| of every layer, and of the SS2D scan output right before its LayerNorm
    (the value that is cast/normalised for fp16; fp16 overflows above 65504), plus AMP GradScaler skips
  * every batch, right after the forward pass: the loss must be finite. Otherwise the same batch is re-run with a
    finiteness check after every layer, the first layer that produced inf/NaN is reported, and the batch
    and weights are saved for offline reproduction. Training then stops.
--legacy reproduces the pre-fix SS2D (scan output cast to fp16 *before* the LayerNorm).

Usage (Kaggle):  python tools/diagnose_nan.py --name diag_fixed
                 python tools/diagnose_nan.py --legacy --name diag_legacy
Output: runs/debug/<name>/{epochs.jsonl, nan_report.json, nan_batch.pt, nan_model.pt}
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FP16_MAX = 65504.0


def tensors(o):
    if hasattr(o, "is_floating_point"):
        yield o
    elif isinstance(o, (list, tuple)):
        for v in o:
            yield from tensors(v)
    elif isinstance(o, dict):
        for v in o.values():
            yield from tensors(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="diag")
    ap.add_argument("--stop-epoch", type=int, default=15, help="stop after this many epochs (schedule stays 300)")
    ap.add_argument("--epochs", type=int, default=300, help="schedule length, as in the real run")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--data", default=None)
    ap.add_argument("--legacy", action="store_true", help="cast the SS2D scan output to fp16 before its LayerNorm")
    args = ap.parse_args()

    os.environ["BLADEYOLO_BOX_LOSS"] = "wiou"
    sys.path.insert(0, ROOT)
    import torch
    import torch.nn.functional as F
    from ultralytics import YOLO

    import models.ss2d as ss2d
    import train
    from models import wiou
    from models.trainer import BladeTrainer

    out = os.path.join(ROOT, "runs", "debug", args.name)
    os.makedirs(out, exist_ok=True)
    st = dict(act={}, prenorm={}, current=None, localising=False, skips=0, scale=None)

    class SS2DFunctional:
        """Stand-in for torch.nn.functional inside models/ss2d.py: records the pre-norm scan output."""

        def __getattr__(self, name):
            return getattr(F, name)

        def layer_norm(self, y, *a, **k):
            if torch.is_grad_enabled() and st["current"]:
                v = y.detach().abs().amax().float()
                p = st["prenorm"]
                p[st["current"]] = torch.maximum(p[st["current"]], v) if st["current"] in p else v
            if args.legacy:
                y = y.to(torch.float16).float()  # old code: y.to(z.dtype) before the norm
            return F.layer_norm(y, *a, **k)

    ss2d.F = SS2DFunctional()
    ss2d_forward = ss2d.SS2D.forward

    def named_ss2d_forward(self, x):
        st["current"] = getattr(self, "_diag_name", None)
        return ss2d_forward(self, x)

    ss2d.SS2D.forward = named_ss2d_forward

    def on_train_start(tr):
        m = tr.model
        for name, mod in m.named_modules():
            if isinstance(mod, ss2d.SS2D):
                mod._diag_name = name
            if name and not any(True for _ in mod.children()):

                def hook(_, __, o, name=name):
                    if not torch.is_grad_enabled():
                        return
                    for t in tensors(o):
                        if t.is_floating_point() and t.numel():
                            v = t.detach().abs().amax().float()
                            st["act"][name] = torch.maximum(st["act"][name], v) if name in st["act"] else v

                mod.register_forward_hook(hook)

        def check_loss(_, a, o):
            """Right after the forward pass, before backward/step: weights are exactly those that failed."""
            if st["localising"] or not torch.is_grad_enabled() or not isinstance(a[0], dict):
                return
            if not torch.isfinite(o[0]).all():
                localise(tr, a[0])

        m.register_forward_hook(check_loss)
        print(f"[diag] monitoring all layers; legacy SS2D={args.legacy}")

    def localise(tr, b):
        """Re-run the failing batch with a finiteness check after every layer."""
        st["localising"] = True
        m, trace = tr.model, []

        def check(mod, i, o, name):
            outs = [t for t in tensors(o) if t.is_floating_point()]
            ins = [t for t in tensors(i) if t.is_floating_point()]
            trace.append(dict(
                layer=name, type=type(mod).__name__,
                out_finite=all(bool(torch.isfinite(t).all()) for t in outs),
                in_finite=all(bool(torch.isfinite(t).all()) for t in ins),
                out_absmax=max((float(t.detach().float().abs().amax()) for t in outs), default=0.0),
                in_absmax=max((float(t.detach().float().abs().amax()) for t in ins), default=0.0),
                out_dtype=str(outs[0].dtype) if outs else None,
            ))

        hs = [mod.register_forward_hook(lambda mod, i, o, name=name: check(mod, i, o, name)) for name, mod in m.named_modules() if name]
        with torch.no_grad(), torch.autocast("cuda", enabled=bool(tr.amp)):  # values only, no second graph
            loss, items = m(b)
        for h in hs:
            h.remove()
        first = next((t for t in trace if not t["out_finite"]), None)
        report = dict(
            epoch=tr.epoch + 1,
            legacy_ss2d=args.legacy,
            loss=[float(v) for v in loss.detach().float().flatten()],
            first_nonfinite_layer=first,
            layers_before_it=trace[max(0, trace.index(first) - 5): trace.index(first)] if first else None,
            verdict=("forward pass produced inf/NaN" if first else "all layer outputs finite: the loss computation itself is non-finite"),
            wiou_running_mean=None if wiou._RunningMean.value is None else float(wiou._RunningMean.value),
            amp_scale=tr.scaler.get_scale() if tr.amp else None,
        )
        json.dump(report, open(os.path.join(out, "nan_report.json"), "w"), indent=2)
        torch.save({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in b.items()}, os.path.join(out, "nan_batch.pt"))
        torch.save(m.state_dict(), os.path.join(out, "nan_model.pt"))
        print("\n[diag] NON-FINITE LOSS\n" + json.dumps(report, indent=2))
        raise RuntimeError(f"[diag] stopped at first non-finite loss; see {out}/nan_report.json")

    def on_train_batch_end(tr):
        if tr.amp:
            s = tr.scaler.get_scale()
            if st["scale"] is not None and s < st["scale"]:
                st["skips"] += 1  # GradScaler lowered the scale: that step had inf/NaN gradients and was skipped
            st["scale"] = s

    def on_fit_epoch_end(tr):
        if st.get("done"):  # Ultralytics calls this once more during the final evaluation
            return
        act = {k: float(v) for k, v in st["act"].items()}
        pre = {k: float(v) for k, v in st["prenorm"].items()}
        top = sorted(act.items(), key=lambda kv: -kv[1])[:8]
        row = dict(
            epoch=tr.epoch + 1,
            losses={k: float(v) for k, v in (tr.tloss or {}).items()},
            val_mAP50=tr.metrics.get("metrics/mAP50(B)"),
            amp_scale=st["scale"],
            amp_skipped_steps=st["skips"],
            ss2d_prenorm_absmax=pre,
            ss2d_prenorm_over_fp16=[k for k, v in pre.items() if v > FP16_MAX],
            top_activations=top,
            layers_over_half_fp16=sum(v > FP16_MAX / 2 for v in act.values()),
        )
        with open(os.path.join(out, "epochs.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"\n[diag] epoch {row['epoch']}: SS2D pre-norm max {max(pre.values(), default=0):.4g} "
              f"(fp16 max {FP16_MAX:.0f}), largest layer {top[0][0] if top else '-'}={top[0][1] if top else 0:.4g}, "
              f"AMP skipped steps {st['skips']}")
        st["act"].clear(), st["prenorm"].clear()
        st["skips"] = 0
        if tr.epoch + 1 >= args.stop_epoch:
            tr.stop = st["done"] = True

    model = YOLO(os.path.join(ROOT, "bladeyolo-l.yaml"))
    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.train(
        trainer=BladeTrainer,
        data=train.resolve_data(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=640,
        device=args.device,
        workers=4,
        cache="ram",
        amp=True,
        project=os.path.join(ROOT, "runs", "debug"),
        name=args.name,
        exist_ok=True,
        **train.RECIPE,
    )


if __name__ == "__main__":
    main()
