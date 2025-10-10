"""
GPU detection and configuration utilities
"""

import os
import subprocess


def detect_amd_gpu() -> bool:
    """
    Detect if an AMD GPU is present in the system.

    Returns:
        True if AMD GPU detected, False otherwise
    """
    try:
        # Check for rocm-smi (ROCm System Management Interface)
        result = subprocess.run(['rocm-smi'], capture_output=True, text=True, timeout=2)
        if result.returncode == 0 and 'AMD' in result.stdout:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    try:
        # Check lspci for AMD GPU
        result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=2)
        if 'VGA compatible controller: Advanced Micro Devices' in result.stdout or \
           'Display controller: Advanced Micro Devices' in result.stdout:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return False


def configure_amd_gpu() -> None:
    """
    Configure AMD GPU environment variables if an AMD GPU is detected.
    Must be called before importing torch.
    """
    if detect_amd_gpu():
        print("AMD GPU detected - configuring ROCm environment variables")
        # PyTorch is compiled for gfx1100, not gfx1101
        # Override to use gfx1100 kernels for gfx1101 GPU (RX 7700/7800 XT)
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
        os.environ["HIP_VISIBLE_DEVICES"] = "0"
        os.environ["AMD_SERIALIZE_KERNEL"] = "3"
        os.environ["TORCH_USE_HIP_DSA"] = "1"
        # Force use of gfx1100 architecture (closest match in PyTorch)
        os.environ["PYTORCH_ROCM_ARCH"] = "gfx1100"
        # Enable expandable memory segments for better memory management
        os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
        # Disable async kernel launches for better error reporting
        os.environ["HIP_LAUNCH_BLOCKING"] = "1"

        print(f"  HSA_OVERRIDE_GFX_VERSION: {os.environ['HSA_OVERRIDE_GFX_VERSION']}")
        print(f"  PYTORCH_ROCM_ARCH: {os.environ['PYTORCH_ROCM_ARCH']}")
        print(f"  TORCH_USE_HIP_DSA: {os.environ['TORCH_USE_HIP_DSA']}")
        print(f"  HIP_LAUNCH_BLOCKING: {os.environ['HIP_LAUNCH_BLOCKING']}")
