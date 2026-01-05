import argparse
import sys
import os
import torch
import torch.nn as nn
from model import Network_woCalibrate

def optimize_model(ckpt_path, output_dir="weights/optimized"):
    print(f"Loading checkpoint: {ckpt_path}")
    
    # 1. Setup
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Model
    model = Network_woCalibrate()
    try:
        weights_dict = torch.load(ckpt_path, map_location=device)
        model_dict = model.state_dict()
        weights_dict = {k: v for k, v in weights_dict.items() if k in model_dict}
        model_dict.update(weights_dict)
        model.load_state_dict(model_dict)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.to(device)
    model.eval()
    
    # Dummy Inputs for Tracing/Export
    dummy_img = torch.randn(1, 3, 480, 640).to(device)
    dummy_sem = torch.randn(1, 3, 480, 640).to(device)
    dummy_depth = torch.randn(1, 3, 480, 640).to(device)

    print("Model loaded. Starting optimization...")

    # ---------------------------------------------------------
    # 2. TorchScript Tracing (JIT)
    # ---------------------------------------------------------
    print("\n[1/2] creating TorchScript (Traced) model...")
    try:
        traced_model = torch.jit.trace(model, (dummy_img, dummy_sem, dummy_depth))
        jit_path = os.path.join(output_dir, "model_traced.pt")
        torch.jit.save(traced_model, jit_path)
        print(f"Saved: {jit_path}")
    except Exception as e:
        print(f"Failed to trace model: {e}")

    # ---------------------------------------------------------
    # 3. FP16 (Half Precision)
    # ---------------------------------------------------------
    if device.type == 'cuda':
        print("\n[2/2] Transforming to FP16 (Half Precision)...")
        try:
            model_half = model.half()
            half_inputs = (dummy_img.half(), dummy_sem.half(), dummy_depth.half())
            
            # Verify run
            with torch.no_grad():
                _ = model_half(*half_inputs)
                
            fp16_path = os.path.join(output_dir, "model_fp16.pt")
            torch.save(model_half.state_dict(), fp16_path)
            print(f"Saved weights: {fp16_path}")
            
            # Optional: Trace FP16
            traced_half = torch.jit.trace(model_half, half_inputs)
            jit_half_path = os.path.join(output_dir, "model_fp16_traced.pt")
            torch.jit.save(traced_half, jit_half_path)
            print(f"Saved Traced FP16: {jit_half_path}")
            
        except Exception as e:
            print(f"Failed FP16: {e}")
    else:
        print("\n[2/2] Skipping FP16 (CUDA not available)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="weights/final/my_model.pt", help="Path to input checkpoint")
    args = parser.parse_args()
    
    optimize_model(args.path)
