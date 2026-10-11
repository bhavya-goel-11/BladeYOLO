"""Keep a block in fp32 under mixed precision."""

import torch


class FP32Block:
    """Mixin: the block computes in fp32 and keeps fp32 weights, whatever the surrounding precision.

    Under AMP autocast is disabled inside the block; when the model is converted with .half() (Ultralytics does this
    for validation and checkpoints) the block's tensors are converted back to fp32. Inputs are cast to fp32 and the
    output back to the input dtype, so the rest of the network is unaffected.
    """

    def _apply(self, fn, *args, **kwargs):
        super()._apply(fn, *args, **kwargs)
        if any(p.dtype in (torch.float16, torch.bfloat16) for p in self.parameters()):
            super()._apply(lambda t: t.float() if t.is_floating_point() else t)
        return self

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return super().forward(x.float()).to(x.dtype)
