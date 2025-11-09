import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from fastsam import FastSAM, FastSAMPrompt

def parse_args():
    """Parses command-line arguments for batch processing."""
    parser = argparse.ArgumentParser(description="Batch process LOL-v2 dataset with FastSAM (Automatic Masking Only)")
    
    # --- Core Arguments ---
    parser.add_argument(
        "--model_path", type=str, default="./weights/FastSAM.pt", help="Path to the FastSAM model weights"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output/", help="Directory to save segmented masks"
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split to process (e.g., 'train')"
    )
    
    # --- Model Config Arguments ---
    parser.add_argument(
        "--imgsz", type=int, default=1024, help="Image size for the model"
    )
    parser.add_argument(
        "--iou", type=float, default=0.9, help="IOU threshold for filtering annotations"
    )
    parser.add_argument(
        "--conf", type=float, default=0.4, help="Object confidence threshold"
    )
    
    # --- Output/Visualization Arguments ---
    parser.add_argument(
        "--better_quality", type=bool, default=False, help="Use morphologyEx for better quality"
    )
    parser.add_argument(
        "--retina", type=bool, default=True, help="Draw high-resolution segmentation masks"
    )
    parser.add_argument(
        "--withContours", type=bool, default=False, help="Draw the edges of the masks"
    )
    
    # --- Device Argument ---
    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
    parser.add_argument(
        "--device", type=str, default=device, help="Device to run on (e.g., 'cuda', 'cpu')"
    )
    
    # Note: All prompt-based arguments (text_prompt, box_prompt, point_prompt) 
    # have been removed to keep this script simple and automatic.
    
    return parser.parse_args()

def run_automatic_segmentation(model, input_image, output_path, args):
    """
    Runs FastSAM "everything" segmentation on a single PIL image and saves the result.
    """
    # 1. Run the model to get all possible segmentations
    everything_results = model(
        input_image,
        device=args.device,
        retina_masks=args.retina,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou
    )
    
    # 2. Initialize the prompt processor
    prompt_process = FastSAMPrompt(input_image, everything_results, device=args.device)
    
    # 3. This is the key part: We ONLY call everything_prompt() for automatic masks
    ann = prompt_process.everything_prompt()

    # --- START DEBUGGING ---
    print(f"\n[Debug] Attempting to save to: {output_path}")
    if ann is None or ann.numel() == 0:
        print("[Debug] !!! WARNING: `everything_prompt()` returned no annotations. Nothing to plot.")
        return  # Exit function, nothing to save
    
    print(f"[Debug] Found {len(ann)} annotations to plot.")
    # --- END DEBUGGING ---

    # 4. Plot and save the fully segmented image
    try:
        prompt_process.plot(
            annotations=ann,
            output_path=output_path,
            bboxes=None,
            points=None,
            point_label=None,
            withContours=args.withContours,
            better_quality=args.better_quality,
        )

        # --- START DEBUGGING 2 ---
        if os.path.exists(output_path):
            print(f"[Debug] Successfully saved: {output_path}")
        else:
            print(f"[Debug] !!! FAILED TO SAVE: Plot function ran but did not create file.")
        # --- END DEBUGGING 2 ---

    except Exception as e:
        # --- START DEBUGGING 3 ---
        print(f"\n[Debug] !!! ERROR during plotting or saving: {e}")
        import traceback
        traceback.print_exc()
        # --- END DEBUGGING 3 ---

def main(args):
    """
    Main function to load model, dataset, and process images.
    """
    print(f"Loading model from {args.model_path}...")
    model = FastSAM(args.model_path)
    model.to(args.device)
    print(f"Using device: {args.device}")

    print("Loading dataset 'okhater/lolv2-real'...")
    try:
        # You must be logged in: `huggingface-cli login`
        ds = load_dataset("okhater/lolv2-real", data_files="Train/*/0012*")
    except Exception as e:
        print(f"Failed to load dataset. Make sure you are logged in to Hugging Face CLI.")
        print("Run: `huggingface-cli login` in your terminal.")
        print(f"Error: {e}")
        return

    # Check if the specified split exists
    if args.split not in ds:
        print(f"Error: Split '{args.split}' not found. Available splits: {list(ds.keys())}")
        return

    split_data = ds[args.split]
    
    # Filter for low-light images (label index 0)
    target_label_index = 0
    print(f"Filtering for low-light images (label index: {target_label_index})")
    try:
        low_light_ds = split_data.filter(lambda example: example['label'] == target_label_index)
        print(f"Found {len(low_light_ds)} low-light images to process.")
    except Exception as e:
        print(f"Failed to filter dataset. Error: {e}")
        return
        
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving masks to {args.output_dir}")

    # Loop over all found low-light images and process them
    for i, item in enumerate(tqdm(low_light_ds, desc=f"Processing {args.split} split")):
        input_image = item['image'].convert("RGB")
        
        # Create a unique filename for each output image
        output_filename = f"{args.split}_low_{i:05d}.png"
        output_path = os.path.join(args.output_dir, output_filename)
        
        try:
            run_automatic_segmentation(model, input_image, output_path, args)
        except Exception as e:
            print(f"\nWarning: Failed to process image {i}. Error: {e}")
            # Continue to the next image
            
    print(f"\nProcessing complete. All masks saved to {args.output_dir}")

if __name__ == "__main__":
    args = parse_args()
    main(args)