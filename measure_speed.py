import os
import sys

# --- ADD THESE TWO LINES ---
FASTSAM_DIR = os.path.join(os.path.dirname(__file__), 'FastSAM')
sys.path.insert(0, FASTSAM_DIR)
# ---------------------------

import torch
import argparse
import numpy as np
import time
from PIL import Image
from tqdm import tqdm

# --- Model Imports ---
# 1. From your test.py and model.py
from model import Network_woCalibrate 
# 2. From your fastsam.py
from FastSAM.fastsam import FastSAM, FastSAMPrompt 
# 3. From your depth-anything.py
from transformers import pipeline
# 4. For data transforms
from torchvision.transforms.functional import to_tensor

# --- Argument Parsing ---
parser = argparse.ArgumentParser("Enlighten-Anything End-to-End Speed Test")
parser.add_argument('--test_dir', type=str, default='./data/lolv2-real/train/low', help='Directory of raw test images')
parser.add_argument('--enhance_weights', type=str, default="weights/final/weights_999.pt", help='Path to your EnhanceNetwork weights')
parser.add_argument('--fastsam_weights', type=str, default="weights/FastSAM-s.pt", help='Path to your FastSAM model weights')
parser.add_argument('--depth_model_name', type=str, default="depth-anything/Depth-Anything-V2-Small-hf", help='HuggingFace name of the depth model')
parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
parser.add_argument('--seed', type=int, default=2, help='Random seed')
args = parser.parse_args()

def load_models(device):
    """
    Loads all three models into memory and sets them to evaluation mode.
    """
    print(f"Loading all models onto device: {device}")
    
    # 1. Load EnhanceNetwork (from test.py)
    enhance_model = Network_woCalibrate()
    weights_dict = torch.load(args.enhance_weights, map_location=device)
    model_dict = enhance_model.state_dict()
    weights_dict = {k: v for k, v in weights_dict.items() if k in model_dict}
    model_dict.update(weights_dict)
    enhance_model.load_state_dict(model_dict)
    enhance_model.to(device)
    enhance_model.eval()
    print("  ... EnhanceNetwork loaded.")

    # 2. Load FastSAM (from fastsam.py)
    fastsam_model = FastSAM(args.fastsam_weights)
    fastsam_model.to(device)
    # fastsam_model.eval()
    print("  ... FastSAM loaded.")

    # 3. Load Depth-Anything (from depth-anything.py)
    # Note: pipeline() handles moving the model to the correct device (gpu_id)
    depth_pipe = pipeline(task="depth-estimation", model=args.depth_model_name, device=device)
    print("  ... Depth-Anything loaded.")
    
    return enhance_model, fastsam_model, depth_pipe

def process_sem_input(fastsam_model, image_pil, device):
    """
    Runs FastSAM on a PIL image and processes the output masks
    into a 3-channel tensor as required by the EnhanceNetwork.
    
    This replaces saving a plotted image to disk.
    """
    # Run FastSAM model
    everything_results = fastsam_model(
        image_pil,
        device=device,
        retina_masks=True,
        imgsz=1024, # Standard FastSAM param,
        conf=0.4,
        iou=0.9
    )
    
    # Process results
    prompt_process = FastSAMPrompt(image_pil, everything_results, device=device)
    
    # ann can be a Tensor (if objects are found) or a list (if not)
    ann = prompt_process.everything_prompt() 
    
    ann_tensor = None

    # --- NEW ROBUST CHECK ---
    if isinstance(ann, torch.Tensor):
        # Case 1: Objects were found, ann is a (N, H, W) tensor
        if ann.numel() == 0:
            # Tensor is empty (e.g., shape [0, H, W])
            pass # Will be handled by the 'ann_tensor is None' check
        else:
            ann_tensor = ann

    elif isinstance(ann, list):
        # Case 2: No objects found (ann is []) or it's a list of tensors
        if not ann:
            # List is empty
            pass # Will be handled by the 'ann_tensor is None' check
        else:
            # It's a list of tensors, stack them
            try:
                ann_tensor = torch.stack(ann)
            except Exception as e:
                print(f"Warning: Could not stack masks for an image: {e}.")
                pass # Will be handled by the 'ann_tensor is None' check
    
    # --- END NEW CHECK ---

    # If no valid masks were found in any case, return a blank tensor
    if ann_tensor is None or ann_tensor.numel() == 0:
        H, W = image_pil.height, image_pil.width
        return torch.zeros(1, 3, H, W, device=device, dtype=torch.float)

    # We now have a valid (N, H, W) tensor, merge all masks
    sem_mask, _ = torch.max(ann_tensor, dim=0) # Result is (H, W)
    
    # Convert (H, W) to (1, 3, H, W) for the network
    sem_tensor = sem_mask.float().unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
    
    return sem_tensor

def process_depth_input(depth_pipe, image_pil, device):
    """
    Runs Depth-Anything on a PIL image and processes the output
    into a 3-channel tensor.
    """
    output = depth_pipe(image_pil)
    depth_pil = output["depth"] # This is a single-channel (L) PIL image
    
    # Convert (L) to (RGB) and then to a [0,1] tensor
    depth_pil_rgb = depth_pil.convert("RGB")
    depth_tensor = to_tensor(depth_pil_rgb).unsqueeze(0).to(device)
    
    return depth_tensor

def tensor_to_pil(t):
    t = t.squeeze(0).detach().cpu().clamp(0, 1)
    return Image.fromarray((t.permute(1, 2, 0).numpy() * 255).astype('uint8'))

def main():
    if not torch.cuda.is_available():
        print('No GPU device available')
        sys.exit(1)
        
    device = torch.device(f"cuda:{args.gpu}")
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # --- 1. Load Models ---
    enhance_model, fastsam_model, depth_pipe = load_models(device)
    
    # --- 2. Get Test Images ---
    image_list = [
        f for f in os.listdir(args.test_dir) 
        if f.endswith('.jpg') or f.endswith('.png')
    ]
    if not image_list:
        print(f"No test images found in {args.test_dir}. Exiting.")
        sys.exit(1)
        
    print(f"\nFound {len(image_list)} images for testing in {args.test_dir}")

    output_dir = "./results_enhanced"
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 3. GPU Warm-up ---
    print("Running GPU warm-up...")
    with torch.no_grad():
        dummy_pil = Image.new('RGB', (640, 480))
        # Create a dummy input tensor
        in_tensor = to_tensor(dummy_pil).unsqueeze(0).to(device)
        # Run FastSAM
        sem_tensor = process_sem_input(fastsam_model, dummy_pil, device)
        # Run Depth
        depth_tensor = process_depth_input(depth_pipe, dummy_pil, device)
        # Run Enhance
        _ = enhance_model(in_tensor, sem_tensor, depth_tensor)
        
    torch.cuda.synchronize()
    
    # --- 4. Run Timed Inference ---
    print("Starting timed inference run...")
    
    # Use CUDA Events for accurate GPU timing
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    total_time_ms = 0.0
    
    with torch.no_grad():
        for img_name in tqdm(image_list, desc="Processing Images"):
            img_path = os.path.join(args.test_dir, img_name)
            image_pil = Image.open(img_path).convert("RGB")
            
            # --- Start Timer ---
            starter.record()
            
            # 1. Process Input Image
            in_tensor = to_tensor(image_pil).unsqueeze(0).to(device)
            
            # 2. Run FastSAM
            sem_tensor = process_sem_input(fastsam_model, image_pil, device)
            
            # 3. Run Depth-Anything
            depth_tensor = process_depth_input(depth_pipe, image_pil, device)
            
            # 4. Run EnhanceNetwork
            # This is the full pipeline from your test.py
            i, r, d = enhance_model(in_tensor, sem_tensor, depth_tensor)
            
            out_pil = tensor_to_pil(r)
            save_path = os.path.join(output_dir, img_name.replace(".jpg", "_enh.jpg").replace(".png", "_enh.png"))
            out_pil.save(save_path)
            # --- Stop Timer ---
            ender.record()
            
            # Wait for all GPU operations to finish
            torch.cuda.synchronize()
            
            # Add time in milliseconds
            total_time_ms += starter.elapsed_time(ender)
    
    # --- 5. Calculate and Print Results ---
    avg_time_ms = total_time_ms / len(image_list)
    avg_fps = 1000.0 / avg_time_ms
    
    print("\n--- Inference Speed Results ---")
    print(f"Total Images Processed: {len(image_list)}")
    print(f"Total Time Taken:       {total_time_ms / 1000.0:.2f} seconds")
    print(f"Average Inference Time: {avg_time_ms:.2f} ms per image")
    print(f"Average FPS (End-to-End): {avg_fps:.2f} FPS")
    print("---------------------------------")


if __name__ == '__main__':
    main()