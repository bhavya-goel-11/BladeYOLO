import sys
from unittest.mock import MagicMock

class MockModule(MagicMock):
    pass

sys.modules['torch'] = MockModule()
sys.modules['torch.nn'] = MockModule()
sys.modules['torch.nn.functional'] = MockModule()
sys.modules['torch.nn.init'] = MockModule()
sys.modules['torch.fft'] = MockModule()
sys.modules['dinov3'] = MockModule()
sys.modules['dinov3.models'] = MockModule()
sys.modules['dinov3.models.vision_transformer'] = MockModule()
sys.modules['dinov3.utils'] = MockModule()
sys.modules['dinov3.layers'] = MockModule()
sys.modules['dinov3.layers.attention'] = MockModule()
sys.modules['dinov3.layers.mlp'] = MockModule()
sys.modules['dinov3.layers.layer_scale'] = MockModule()

sys.modules['ultralytics'] = MockModule()
sys.modules['ultralytics.nn'] = MockModule()
sys.modules['ultralytics.nn.modules'] = MockModule()
sys.modules['ultralytics.nn.modules.conv'] = MockModule()
sys.modules['ultralytics.nn.modules.utils'] = MockModule()
sys.modules['ultralytics.nn.modules.block'] = MockModule()
sys.modules['ultralytics.utils'] = MockModule()
sys.modules['ultralytics.utils.torch_utils'] = MockModule()
sys.modules['ultralytics.nn.tasks'] = MockModule()
sys.modules['pywt'] = MockModule()
sys.modules['einops'] = MockModule()
sys.modules['mamba_ssm'] = MockModule()
sys.modules['vmamba'] = MockModule()

try:
    import train
    print("ALL IMPORTS RESOLVED SUCCESSFULLY!")
except Exception as e:
    import traceback
    print("IMPORT ERROR FOUND:")
    traceback.print_exc()
