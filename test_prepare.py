import torch
from dinov3.models.vision_transformer import vit_small
model = vit_small()
x = torch.zeros(1, 3, 224, 224)
out = model.prepare_tokens_with_masks(x)
print("Type of out:", type(out))
if isinstance(out, tuple):
    print("Tuple length:", len(out))
    for i, item in enumerate(out):
        print(f"Item {i} type:", type(item))
elif isinstance(out, dict):
    print("Dict keys:", out.keys())
