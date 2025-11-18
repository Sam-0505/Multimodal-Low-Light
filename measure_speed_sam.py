import os
import sys
import torch
import argparse
import numpy as np
import time
from PIL import Image
from tqdm import tqdm
import cv2 # Required for SAM's input format

# --- Model Imports ---
# 1. From your test.py and model.py
from model import Network_woCalibrate 
# 2. From your sam.py
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
# 3. From your depth-anything.py
from transformers import pipeline
# 4. For data transforms
from torchvision.transforms.functional import to_tensor

# --- Argument Parsing ---
parser = argparse.ArgumentParser("Enlighten-Anything End-to-End Speed Test (using SAM)")
parser.add_argument('--test_dir', type=str, default='data/LOL/test15/low', help='Directory of raw test images')
parser.add_argument('--enhance_weights', type=str, default=".\checkpoints\Train-20251117-131243\model_epochs\weights_0.pt", help='Path to your EnhanceNetwork weights')
parser.add_argument('--sam_weights', type=str, default="segment_anything/sam_vit_h_4b8939.pth", help='Path to your SAM model weights')
parser.add_argument('--sam_model_type', type=str, default="vit_h", help='SAM model type (e.g., vit_h, vit_b)')
parser.add_argument('--depth_model_name', type=str, default="LiheYoung/depth-anything-large-hf", help='HuggingFace name of the depth model')
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

    # 2. Load SAM (from sam.py)
    print("  ... Loading SAM...")
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_weights)
    sam.to(device=device)
    sam.eval()
    # This generator is what we use for inference
    mask_generator = SamAutomaticMaskGenerator(sam)
    print("  ... SAM loaded.")

    # 3. Load Depth-Anything (from depth-anything.py)
    depth_pipe = pipeline(task="depth-estimation", model=args.depth_model_name, device=device)
    print("  ... Depth-Anything loaded.")
    
    return enhance_model, mask_generator, depth_pipe

def process_sem_input_sam(mask_generator, image_pil, device):
    """
    Runs SAM on a PIL image and processes the output masks
    into a 3-channel tensor as required by the EnhanceNetwork.
    
    SAM's mask_generator expects a (H, W, 3) numpy array in RGB format.
    """
    # 1. Convert PIL image to (H, W, 3) RGB numpy array
    image_np_rgb = np.array(image_pil)
    
    # 2. Run SAM Mask Generator
    # This is the main inference step for SAM
    masks = mask_generator.generate(image_np_rgb)
    
    # 3. Process masks
    if not masks:
        # If no masks are found, return a blank tensor
        H, W, _ = image_np_rgb.shape
        return torch.zeros(1, 3, H, W, device=device, dtype=torch.float)

    # 4. Merge all masks into one single (H, W) boolean mask
    # We take the logical OR of all segmentation masks
    H, W = masks[0]['segmentation'].shape
    merged_mask_np = np.zeros((H, W), dtype=bool)
    for mask_dict in masks:
        merged_mask_np |= mask_dict['segmentation']
        
    # 5. Convert the (H, W) numpy mask to a (1, 3, H, W) torch tensor
    merged_mask_tensor = torch.from_numpy(merged_mask_np).float().to(device)
    sem_tensor = merged_mask_tensor.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
    
    return sem_tensor

def process_depth_input(depth_pipe, image_pil, device):
    """
    Runs Depth-Anything on a PIL image and processes the output
    into a 3-channel tensor. (Unchanged from previous script)
    """
    output = depth_pipe(image_pil)
    depth_pil = output["depth"] # This is a single-channel (L) PIL image
    
    # Convert (L) to (RGB) and then to a [0,1] tensor
    depth_pil_rgb = depth_pil.convert("RGB")
    depth_tensor = to_tensor(depth_pil_rgb).unsqueeze(0).to(device)
    
    return depth_tensor

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
    enhance_model, mask_generator, depth_pipe = load_models(device)
    
    # --- 2. Get Test Images ---
    image_list = [
        f for f in os.listdir(args.test_dir) 
        if f.endswith('.jpg') or f.endswith('.png')
    ]
    if not image_list:
        print(f"No test images found in {args.test_dir}. Exiting.")
        sys.exit(1)
        
    print(f"\nFound {len(image_list)} images for testing in {args.test_dir}")
    
    # --- 3. GPU Warm-up ---
    print("Running GPU warm-up...")
    with torch.no_grad():
        dummy_pil = Image.new('RGB', (640, 480))
        # Create a dummy input tensor
        in_tensor = to_tensor(dummy_pil).unsqueeze(0).to(device)
        # Run SAM
        sem_tensor = process_sem_input_sam(mask_generator, dummy_pil, device)
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
            
            # 2. Run SAM
            sem_tensor = process_sem_input_sam(mask_generator, image_pil, device)
            
            # 3. Run Depth-Anything
            depth_tensor = process_depth_input(depth_pipe, image_pil, device)
            
            # 4. Run EnhanceNetwork
            i, r, d = enhance_model(in_tensor, sem_tensor, depth_tensor)
            
            # --- Stop Timer ---
            ender.record()
            
            # Wait for all GPU operations to finish
            torch.cuda.synchronize()
            
            # Add time in milliseconds
            total_time_ms += starter.elapsed_time(ender)
    
    # --- 5. Calculate and Print Results ---
    avg_time_ms = total_time_ms / len(image_list)
    avg_fps = 1000.0 / avg_time_ms
    
    print("\n--- Inference Speed Results (with SAM) ---")
    print(f"Total Images Processed: {len(image_list)}")
    print(f"Total Time Taken:       {total_time_ms / 1000.0:.2f} seconds")
    print(f"Average Inference Time: {avg_time_ms:.2f} ms per image")
    print(f"Average FPS (End-to-End): {avg_fps:.2f} FPS")
    print("------------------------------------------")


if __name__ == '__main__':
    main()