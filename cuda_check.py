#!/usr/bin/env python3
# cuda_check.py
# Script to diagnose CUDA availability for PyTorch

import torch
import sys
import subprocess

def check_cuda():
    print("=== PyTorch CUDA Information ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Device count: {torch.cuda.device_count()}")
    else:
        print("\nCUDA is not available for PyTorch!")
        print("\nPossible reasons:")
        print("1. PyTorch was installed without CUDA support")
        print("2. NVIDIA drivers are not properly installed")
        print("3. Your GPU is not CUDA compatible")
        print("4. CUDA version mismatch between PyTorch and your system")
    
    print("\n=== System GPU Information ===")
    try:
        if sys.platform == 'win32':
            # Windows
            nvidia_smi = subprocess.check_output('nvidia-smi', shell=True)
        else:
            # Linux/Mac
            nvidia_smi = subprocess.check_output(['nvidia-smi'])
        print(nvidia_smi.decode('utf-8'))
    except:
        print("nvidia-smi command failed - GPU may not be accessible")
    
    print("\n=== Reinstallation Instructions ===")
    print("To reinstall PyTorch with CUDA support, please visit:")
    print("https://pytorch.org/get-started/locally/")
    print("\nFor example, to install PyTorch with CUDA 11.8 support using pip:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

if __name__ == "__main__":
    check_cuda() 