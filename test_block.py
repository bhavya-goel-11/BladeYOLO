import inspect
from dinov3.models.vision_transformer import vit_small
model = vit_small()
print(inspect.signature(model.blocks[0].forward))
