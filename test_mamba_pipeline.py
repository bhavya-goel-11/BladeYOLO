import sys
import os
import torch
import torch.nn as nn

# Ensure the script can find the local BladeYOLO modules regardless of where it's run from
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def run_tests():
    print("=" * 60)
    print("🧪 BLADEYOLO CROSS-MAMBA END-TO-END TEST SUITE")
    print("=" * 60)

    # 1. Environment Check
    print("\n[1] Checking Environment...")
    try:
        import mamba_ssm
        print("  ✅ mamba_ssm is installed (CUDA acceleration available)")
    except ImportError:
        print("  ⚠️ mamba_ssm NOT found! Will use pure-PyTorch fallback (slower).")

    # 2. Basic Import Check
    print("\n[2] Checking Local Module Imports...")
    try:
        from vmamba.models.vssm import CrossSS2D, LSBlock
        from models.cross_mamba import CrossVSSBlock, CrossScaleStateBlock
        print("  ✅ Successfully imported local vmamba and cross_mamba modules.")
    except Exception as e:
        print(f"  ❌ Import Failed: {e}")
        return

    # 3. Component Test: LSBlock
    print("\n[3] Testing LSBlock Forward Pass...")
    try:
        ls = LSBlock(in_features=384)
        x = torch.randn(2, 384, 20, 20)
        out = ls(x)
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
        print(f"  ✅ LSBlock successful. Output shape: {out.shape}")
    except Exception as e:
        print(f"  ❌ LSBlock Failed: {e}")

    # 4. Component Test: CrossSS2D (Core Dual-Input Module)
    print("\n[4] Testing CrossSS2D Forward Pass & Gradients...")
    try:
        css2d = CrossSS2D(guide_dim=384, d_model=384, d_state=16, ssm_ratio=2.0)
        x = torch.randn(2, 384, 20, 20, requires_grad=True)
        guide = torch.randn(2, 384, 20, 20, requires_grad=True)
        
        out = css2d(x, guide)
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
        print(f"  ✅ Forward pass successful. Output shape: {out.shape}")
        
        # Check gradient flow
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "No gradient for target feature (x)"
        assert guide.grad is not None, "No gradient for guide feature (guide_x)"
        print("  ✅ Backward pass (gradient flow) successful on both inputs.")
    except Exception as e:
        print(f"  ❌ CrossSS2D Failed: {e}")

    # 5. Pipeline Test: CrossScaleStateBlock (P5 -> P4 -> P3)
    print("\n[5] Testing Full CrossScaleStateBlock Pipeline...")
    try:
        cssb = CrossScaleStateBlock(in_channels=384)
        P3 = torch.randn(2, 384, 80, 80)
        P4 = torch.randn(2, 384, 40, 40)
        P5 = torch.randn(2, 384, 20, 20)
        
        out_P3, out_P4, out_P5 = cssb([P3, P4, P5])
        
        assert out_P3.shape[2:] == P3.shape[2:], "P3 spatial dimensions altered!"
        assert out_P4.shape[2:] == P4.shape[2:], "P4 spatial dimensions altered!"
        assert out_P5.shape == P5.shape, "P5 should remain untouched!"
        
        print(f"  ✅ Input shapes:  P3{tuple(P3.shape)}, P4{tuple(P4.shape)}, P5{tuple(P5.shape)}")
        print(f"  ✅ Output shapes: P3{tuple(out_P3.shape)}, P4{tuple(out_P4.shape)}, P5{tuple(out_P5.shape)}")
        print("  ✅ Progressive semantic guidance pipeline successful.")
    except Exception as e:
        print(f"  ❌ CrossScaleStateBlock Failed: {e}")
        
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()

