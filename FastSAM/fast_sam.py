import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from fastsam import FastSAM, FastSAMPrompt
import sys

def parse_args():
    """Parses command-line arguments for batch processing."""
    parser = argparse.ArgumentParser(description="Batch process LOL-v2 dataset with FastSAM (Automatic Masking Only)")
    
    # --- Core Arguments ---
    parser.add_argument(
        "--model_path", type=str, default="../weights/FastSAM-s.pt", help="Path to the FastSAM model weights"
    )
    parser.add_argument(
        "--output_dir", type=str, default="../data/", help="Base directory to save images and masks"
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

    # --- Debugging ---
    if ann is None or ann.numel() == 0:
        print(f"[Debug] !!! WARNING: `everything_prompt()` returned no annotations for {output_path}. Skipping.")
        return  # Exit function, nothing to save
    
    # --- THIS IS THE CORRECTED LINE ---
    print(f"[Debug] Found {ann.shape[0]} annotations to plot.")
    # --- END CORRECTION ---

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
    except Exception as e:
        print(f"\n[Debug] !!! ERROR during plotting or saving: {e}")
        import traceback
        traceback.print_exc()

def main(args):
    """
    Main function to load model, dataset, and process images.
    """
    print(f"Loading model from {args.model_path}...")
    model = FastSAM(args.model_path)
    model.to(args.device)
    print(f"Using device: {args.device}")

    dataset_names = ["okhater/lolv2-real","okhater/lolv2-synthetic"]
    print(f"Preparing to process datasets: {dataset_names}")

    for dataset_name in dataset_names:
        print(f"\n=======================================================")
        print(f"STARTING DATASET: {dataset_name}")
        print("=======================================================")

        # This will load PIL images and labels
        print(f"Loading '{dataset_name}' (full dataset)...")
        try:
            ds = load_dataset(dataset_name) 
        except Exception as e:
            print(f"Failed to load dataset {dataset_name}. Make sure you are logged in: `huggingface-cli login`")
            print(f"Error: {e}")
            continue # Skip to the next dataset

        print(f"Found splits: {list(ds.keys())}")
        for split_name, split_data in ds.items():
            
            print(f"\n--- Processing split: '{split_name}' ({len(split_data)} images) ---")
            
            # Create output directories based on dataset and split
            dataset_folder_name = dataset_name.split('/')[-1] # "lolv2-real"
            low_dir = os.path.join(args.output_dir, dataset_folder_name, split_name, "low")
            high_dir = os.path.join(args.output_dir, dataset_folder_name, split_name, "high")
            mask_dir = os.path.join(args.output_dir, dataset_folder_name, split_name, "masks")

            os.makedirs(low_dir, exist_ok=True)
            os.makedirs(high_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)
            
            print(f"Saving LOW images to: {low_dir}")
            print(f"Saving HIGH images to: {high_dir}")
            print(f"Saving MASKS to: {mask_dir}")

            # Loop over items (index 'i' is important)
            for i, item in enumerate(tqdm(split_data, desc=f"Processing {dataset_name} {split_name} split")):
                try:
                    # 'item['image']' is now a PIL Image, not a path
                    input_image = item['image'].convert("RGB") 
                    label = item['label'] # 0 or 1
                    
                    # Create a new filename
                    output_filename = f"{split_name}_{i:05d}.png"

                    # Save files based on label
                    if label == 0:
                        # This is a LOW-LIGHT image
                        low_path = os.path.join(low_dir, output_filename)
                        input_image.save(low_path)
                        
                        # Run segmentation and save the mask
                        mask_path = os.path.join(mask_dir, output_filename)
                        run_automatic_segmentation(model, input_image, mask_path, args)
                        
                    elif label == 1:
                        # This is a HIGH-LIGHT (Ground Truth) image
                        high_path = os.path.join(high_dir, output_filename)
                        input_image.save(high_path)

                except Exception as e:
                    print(f"\nWarning: Failed to process item {i}. Error: {e}")
                    # Continue to the next image
    
    print(f"\nProcessing complete for all datasets. All files saved in {args.output_dir}")

if __name__ == "__main__":
    # Add parent directory to Python path to find 'fastsam'
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    args = parse_args()
    main(args)