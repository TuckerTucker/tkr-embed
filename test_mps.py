#!/usr/bin/env python3
"""Test MPS (Metal Performance Shaders) availability and functionality"""

import torch
import sys

def test_mps():
    print("Testing MPS (Metal Performance Shaders) on Apple Silicon...")
    print("-" * 60)

    # Check if MPS is available
    if torch.backends.mps.is_available():
        print("✅ MPS is available on this system")

        # Check if MPS is built
        if torch.backends.mps.is_built():
            print("✅ PyTorch was built with MPS support")
        else:
            print("❌ PyTorch was NOT built with MPS support")
            print("   You may need to upgrade PyTorch: pip install --upgrade torch")
            return False

        # Test basic MPS operations
        try:
            print("\nTesting MPS operations...")

            # Create a tensor on MPS
            device = torch.device("mps")
            x = torch.randn(100, 100, device=device)
            y = torch.randn(100, 100, device=device)

            # Perform operations
            z = torch.matmul(x, y)
            mean = z.mean()

            print(f"✅ Matrix multiplication successful")
            print(f"✅ Mean calculation successful: {mean.item():.4f}")

            # Test moving tensors between devices
            cpu_tensor = torch.randn(10, 10)
            mps_tensor = cpu_tensor.to(device)
            back_to_cpu = mps_tensor.cpu()

            print("✅ Tensor device transfers successful")

            # Memory check
            print(f"\nMPS Device: {device}")
            print(f"MPS Tensor device: {mps_tensor.device}")

            return True

        except Exception as e:
            print(f"❌ MPS operation failed: {e}")
            print("\nThis might be due to:")
            print("1. Incompatible model/operation for MPS")
            print("2. PyTorch version issues")
            print("3. macOS version compatibility")
            return False
    else:
        print("❌ MPS is NOT available on this system")
        print("\nPossible reasons:")
        print("1. Not running on Apple Silicon (M1/M2/M3)")
        print("2. macOS version < 12.3")
        print("3. PyTorch version doesn't support MPS")
        return False

def check_system_info():
    print("\nSystem Information:")
    print("-" * 60)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")

    # Check for CUDA (shouldn't be available on Mac)
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Check for MPS
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")

    # Platform info
    import platform
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")

if __name__ == "__main__":
    check_system_info()
    print("\n" + "=" * 60)
    success = test_mps()
    print("=" * 60)

    if success:
        print("\n🎉 MPS is working correctly! Your model can use Metal GPU acceleration.")
    else:
        print("\n⚠️  MPS is not fully functional. Model will use CPU instead.")