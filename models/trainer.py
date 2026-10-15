import os

import torch
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.torch_utils import unwrap_model

import models  # noqa: F401  (registers BladeYOLO layers and WIoU in this process, incl. DDP workers)


def load_pretrained(model, weights, layer_map):
    """Copy layers of a stock Ultralytics checkpoint (e.g. yolo12l.pt) into `model`, re-indexed by `layer_map`.

    layer_map: [[first, last, offset], ...]: checkpoint layers first..last go to layer index + offset in `model`.
    Every mapped tensor must match by name and shape, except the class-score convs of the Detect head, whose output
    size depends on the number of classes: a silent partial transfer would make comparisons meaningless.
    """
    from ultralytics.nn.modules import Detect
    from ultralytics.nn.tasks import load_checkpoint

    src, _ = load_checkpoint(weights)
    csd = src.float().state_dict()
    # Released YOLOv12 checkpoints carry a conv bias in the attention position encoding (attn.pe) that the current
    # code no longer has. A BatchNorm follows that conv, so the bias folds exactly into its running mean.
    for k in [k for k in csd if k.endswith("attn.pe.conv.bias")]:
        csd[k.removesuffix("conv.bias") + "bn.running_mean"] -= csd.pop(k)
    own = model.state_dict()
    detect = len(model.model) - 1 if isinstance(model.model[-1], Detect) else None
    out, skipped, bad = {}, [], []
    for k, v in csd.items():
        i = int(k.split(".")[1])
        dst = next((i + off for a, b, off in layer_map if a <= i <= b), None)
        if dst is None:
            continue
        k2 = ".".join(["model", str(dst), *k.split(".")[2:]])
        if k2 in own and own[k2].shape == v.shape:
            out[k2] = v
        elif dst == detect and ".cv3." in k2 and k2 in own:  # class-score layers: nc differs
            skipped.append(k2)
        else:
            bad.append(f"{k} -> {k2}")
    if bad:
        raise RuntimeError(f"{weights}: {len(bad)} tensors do not match the model, e.g. {bad[:3]}")
    model.load_state_dict(out, strict=False)
    LOGGER.info(f"[BladeYOLO] Loaded {len(out)} pretrained tensors from {weights} (layer map {layer_map}); "
                f"{len(skipped)} class-score tensors re-initialised for {model.yaml['nc']} classes")
    return out


class BladeTrainer(DetectionTrainer):
    """DetectionTrainer that keeps DINOv3 frozen and initialises pretrained backbones.

    Ultralytics re-enables requires_grad on every parameter not listed in `freeze`, so the DINOv3 weights are named
    there (layer 0 is DinoConv, its ViT lives under `.dino`). For models without them, e.g. a stock
    YOLO baseline, the pattern matches nothing. A model YAML with a `pretrained` section (weights, layer map) is
    initialised from that checkpoint when training starts from the YAML (not when resuming or fine-tuning a .pt).
    BLADEYOLO_ROT90=p adds random quarter turns to the training augmentation (an environment variable, so DDP
    workers get it too).
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        overrides = {**(overrides or {}), "freeze": ["0.dino"]}
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

    def build_dataset(self, img_path, mode="train", batch=None):
        dataset = super().build_dataset(img_path, mode=mode, batch=batch)
        p = float(os.environ.get("BLADEYOLO_ROT90", 0))
        if mode == "train" and p > 0:
            from .augment import add_rot90

            add_rot90(dataset, p)
        return dataset

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        pre = model.yaml.get("pretrained")
        if weights is None and pre:
            load_pretrained(model, pre["weights"], pre["map"])
        return model

    # Non-finite batches. Under fp16 a rare batch can overflow somewhere in the forward pass (the probe in
    # tools/probe_nan.py found one such layer, now kept in fp32 by FP32Block). GradScaler then skips the optimizer
    # step, so the weights stay finite, but the forward pass has already written NaN into the BatchNorm running
    # statistics; the EMA copies them at the next optimizer step and every later validation scores 0 (training loss
    # looks normal, because train-mode BatchNorm uses batch statistics). So: after every forward pass the BatchNorm
    # buffers (and the loss) are checked; a poisoned set is restored from the last finite copy before the next forward
    # pass or EMA update, and the batch is dropped (GradScaler already skips its gradients). The first few such batches are
    # saved with a per-layer report of where the forward pass first became non-finite, in fp16 and in fp32.
    NAN_REPORTS = 3

    def _bn_buffers(self):
        if getattr(self, "_bn", None) is None:
            self._bn = [b for m in unwrap_model(self.model).modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
                        for b in (m.running_mean, m.running_var) if b is not None]
            self._bn_good, self._bad_batches, self._last_batch, self._tloss_prev = None, 0, None, None
        return self._bn

    def _repair_bn(self):
        """Restore poisoned BatchNorm statistics from the last finite copy; refresh that copy otherwise."""
        bufs = self._bn_buffers()
        bn_ok = bool(torch.stack([torch.isfinite(b).all() for b in bufs]).all()) if bufs else True
        loss_ok = self.loss is None or bool(torch.isfinite(self.loss).all())
        if bn_ok and loss_ok:
            self._bn_good = [b.detach().clone() for b in bufs]
            return
        if not bn_ok and self._bn_good is None:
            raise RuntimeError("BatchNorm statistics are non-finite and no finite copy exists")
        self._bad_batches += 1
        LOGGER.warning(f"[BladeYOLO] non-finite {'forward pass' if not bn_ok else 'loss'} (epoch {self.epoch + 1}, "
                       f"bad batch {self._bad_batches} of the run): batch dropped"
                       f"{', BatchNorm statistics restored' if not bn_ok else ''}")
        # the batch counts for nothing: GradScaler skips its gradients, the logged epoch loss excludes it, and
        # Ultralytics' epoch-level NaN recovery (which reloads last.pt) is not triggered by it
        self.loss = torch.zeros_like(self.loss) if self.loss is not None else None
        self.tloss = self._tloss_prev
        self._skip_step = True  # its gradients are already accumulated in .grad: the next optimizer step is skipped
        if not bn_ok:
            for b, g in zip(bufs, self._bn_good):
                b.copy_(g)
        if self._bad_batches <= self.NAN_REPORTS and self._last_batch is not None:
            try:
                self._report_nan(self._last_batch)
            except Exception as e:  # diagnostics must never stop training
                LOGGER.warning(f"[BladeYOLO] non-finite batch report failed: {e}")
            if self._bn_good is not None:
                for b, g in zip(bufs, self._bn_good):  # the diagnostic forward passes ran in train mode
                    b.copy_(g)

    def preprocess_batch(self, batch):
        self._repair_bn()  # catches a bad previous batch that was not followed by an optimizer step
        self._last_batch = {k: v for k, v in batch.items()}  # CPU tensors as loaded, before moving to the GPU
        self._tloss_prev = self.tloss  # running epoch loss before this batch
        return super().preprocess_batch(batch)

    def optimizer_step(self):
        self._repair_bn()  # before the EMA copies the BatchNorm statistics
        if getattr(self, "_skip_step", False):
            # GradScaler would skip non-finite gradients under AMP, but not in fp32 (--no-amp); drop them either way.
            # With gradient accumulation this also drops the good batches accumulated with the bad one.
            self._skip_step = False
            self.optimizer.zero_grad()
            return
        super().optimizer_step()

    def _report_nan(self, raw):
        """Re-run the bad batch layer by layer (train mode, no gradients) in fp16 and fp32; save batch and report."""
        import json

        model = unwrap_model(self.model)
        batch = super().preprocess_batch({k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in raw.items()})
        report = {"epoch": self.epoch + 1, "rank": RANK}
        for name, amp in (("fp16", True), ("fp32", False)):
            stats, hooks = [], []

            def hook(m, _, out, i=None):
                ts = [out] if isinstance(out, torch.Tensor) else [t for t in out if isinstance(t, torch.Tensor)] \
                    if isinstance(out, (list, tuple)) else []
                ts += [t for t in getattr(out, "dino", None) or [] if isinstance(t, torch.Tensor)]
                fin = all(torch.isfinite(t).all() for t in ts)
                amax = max((float(t.detach().float().nan_to_num(0, 0, 0).abs().max()) for t in ts), default=0.0)
                stats.append(dict(layer=m.i, type=m.type, finite=bool(fin), absmax=amax))

            hooks = [m.register_forward_hook(hook) for m in model.model]
            try:
                with torch.no_grad(), torch.autocast(self.device.type, enabled=amp and self.amp):
                    model(batch["img"])
            finally:
                for h in hooks:
                    h.remove()
            bad = next((s for s in stats if not s["finite"]), None)
            report[name] = dict(first_non_finite=bad, layers=stats)
        out = os.path.join(self.save_dir, "nan_batches")
        os.makedirs(out, exist_ok=True)
        stem = os.path.join(out, f"epoch{self.epoch + 1}_rank{max(RANK, 0)}_{self._bad_batches}")
        torch.save(raw, stem + "_batch.pt")
        json.dump(report, open(stem + ".json", "w"), indent=1)
        f16, f32 = report["fp16"]["first_non_finite"], report["fp32"]["first_non_finite"]
        LOGGER.warning(f"[BladeYOLO] non-finite batch: fp16 first non-finite at "
                       f"{'layer %d (%s)' % (f16['layer'], f16['type']) if f16 else 'none (loss or backward only)'}; "
                       f"fp32 {'also non-finite at layer %d' % f32['layer'] if f32 else 'finite (fp16 overflow)'}; "
                       f"saved {stem}.json")

    def validate(self):
        """Validate, unless the model or EMA still holds non-finite state (last resort behind the per-batch repair).
        Reporting NaN fitness makes Ultralytics reload last.pt and redo the epoch (up to 3 times)."""
        self._repair_bn()  # the epoch's last batch is not followed by a preprocess_batch
        if self._non_finite():
            LOGGER.warning("[BladeYOLO] non-finite model/EMA state: skipping validation, recovering from last.pt")
            return self.metrics, float("nan")
        return super().validate()

    def _non_finite(self):
        """True on every rank if any rank holds a NaN/Inf weight or buffer in the model or EMA."""
        import torch.distributed as dist

        bad = False
        for m in (self.ema.ema if self.ema else None, unwrap_model(self.model)):
            if m is not None and not bad:
                bad = any(not torch.isfinite(t).all() for t in m.state_dict().values() if t.is_floating_point())
        if self.world_size > 1:  # every rank must take the same branch: validation reduces metrics across ranks
            flag = torch.tensor(float(bad), device=self.device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            bad = bool(flag.item())
        return bad
