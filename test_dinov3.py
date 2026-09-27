import torch
model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vits16', pretrained=False, trust_repo=True)
import inspect
print(inspect.signature(model.interpolate_pos_encoding))
import inspect
print(inspect.getsource(model.interpolate_pos_encoding))
