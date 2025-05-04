#!/usr/bin/env python3
# cuda_vs_cpu_matmul.py
# Compares the performance of large matrix multiplication on CPU vs GPU (CUDA).

import torch
import time
import sys

# --- Configuration ---
MATRIX_SIZE = 5000 # Dimension of the square matrix (e.g., 5000x5000)
# -------------------

def perform_matmul(device_type: str, matrix_size: int):
    """Creates two matrices on the specified device and measures matmul time."""
    try:
        device = torch.device(device_type)
        print(f"\nSetting up matrices ({matrix_size}x{matrix_size}) on {device.type.upper()}...")
        
        # Create large random matrices directly on the target device
        a = torch.randn(matrix_size, matrix_size, device=device)
        b = torch.randn(matrix_size, matrix_size, device=device)
        
        print(f"Performing matrix multiplication on {device.type.upper()}...")
        
        # --- Time the matrix multiplication ---
        start_time = time.perf_counter()
        
        # Perform the multiplication
        c = torch.matmul(a, b)
        
        # Ensure operation is complete for accurate timing (especially needed for CUDA)
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        end_time = time.perf_counter()
        # ------------------------------------
        
        duration = end_time - start_time
        print(f"Finished {device.type.upper()} multiplication.")
        
        # Clean up memory (optional, but good practice for large tensors)
        del a
        del b
        del c
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
        return duration

    except RuntimeError as e:
        print(f"Error during {device_type} operation: {e}")
        print("This might be due to insufficient memory (RAM or VRAM). Try a smaller MATRIX_SIZE.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred on {device_type}: {e}")
        return None

def main():
    print(f"--- Matrix Multiplication Performance Comparison ---")
    print(f"Using matrix size: {MATRIX_SIZE}x{MATRIX_SIZE}")
    
    # --- CPU Performance ---
    cpu_time = perform_matmul('cpu', MATRIX_SIZE)
    if cpu_time is not None:
        print(f"CPU Time: {cpu_time:.4f} seconds")
    else:
        print("CPU test failed or was skipped due to error.")

    # --- GPU (CUDA) Performance ---
    if torch.cuda.is_available():
        cuda_time = perform_matmul('cuda', MATRIX_SIZE)
        if cuda_time is not None:
            print(f"CUDA Time: {cuda_time:.4f} seconds")
            
            # Calculate and print speedup
            if cpu_time is not None and cuda_time > 0:
                speedup = cpu_time / cuda_time
                print(f"\nCUDA Speedup: {speedup:.2f}x faster than CPU")
            elif cpu_time is None:
                 print("\nCannot calculate speedup because CPU test failed.")
        else:
            print("CUDA test failed or was skipped due to error.")
            
    else:
        print("\nCUDA device not found. Skipping GPU performance test.")

if __name__ == "__main__":
    main()
