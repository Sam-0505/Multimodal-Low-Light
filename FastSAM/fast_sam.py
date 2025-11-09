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
        "--model_path", type=str, default="/scratch/user/sam0505/Multimodal-Low-Light/weights/FastSAM-s.pt", help="Path to the FastSAM model weights"
    )
    parser.add_argument(
        "--output_dir", type=str, default="/scratch/user/sam0505/Multimodal-Low-Light/data/", help="Base directory to save images and masks"
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

    # Define the local paths to the top-level dataset folders
    base_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '/scratch/user/sam0505/Multimodal-Low-Light/data'))
    
    # --- MODIFICATION: Point to each SPLIT directory directly ---
    # Instead of loading 'lolv2-real' and 'lolv2-synthetic' and *hoping*
    # the library finds the splits, we tell it exactly which splits to load.
    dataset_splits_to_process = [
        {
            "dataset_name": "lolv2-real", 
            "split_name": "train", 
            "path": os.path.join(base_data_path, "lolv2-real", "train")
        },
        {
            "dataset_name": "lolv2-real", 
            "split_name": "test", 
            "path": os.path.join(base_data_path, "lolv2-real", "test")
        },
        {
            "dataset_name": "lolv2-synthetic", 
            "split_name": "train", 
            "path": os.path.join(base_data_path, "lolv2-synthetic", "train")
        },
        {
            "dataset_name": "lolv2-synthetic", 
            "split_name": "test", 
            "path": os.path.join(base_data_path, "lolv2-synthetic", "test")
        }
    ]
    
    print(f"Preparing to process {len(dataset_splits_to_process)} dataset splits...")

    # --- MODIFICATION: Loop over the new list of specific splits ---
    for split_info in dataset_splits_to_process:
        dataset_name = split_info["dataset_name"]
        split_name = split_info["split_name"]
        split_path = split_info["path"] # This is the path to the e.g., .../train/ folder
        
        print(f"\n=======================================================")
        print(f"STARTING DATASET: {dataset_name} (Split: {split_name})")
        print(f"Loading from local path: {split_path}")
        print("=======================================================")

        print(f"Loading '{dataset_name}/{split_name}'...")
        try:
            # Load the dataset from the *specific split path*
            # This will correctly interpret 'Input' and 'GT' as labels
            # load_dataset() will return a dict, e.g., {'train': Dataset(...)}
            # We want the data, so we access the default 'train' key.
            split_data = load_dataset(split_path, trust_remote_code=True)["train"] 
            
        except Exception as e:
            print(f"Failed to load dataset {dataset_name}/{split_name} from {split_path}.")
            print(f"Error: {e}")
            print("!!! Does this directory exist? Skipping...")
            continue 

        # --- MODIFICATION: The 'for split_name, split_data in ds.items():' loop is removed ---
        # We are now *inside* the logic that was in that loop,
        # because we loaded the split directly.
        
        print(f"\n--- Processing split: '{split_name}' ({len(split_data)} images) ---")
        
        try:
            # This logic is now correct, as 'label' will refer to 'Input' and 'GT'
            label_names = split_data.features['label'].names
            low_label_index = -1
            high_label_index = -1
            
            # Loop and find *both* labels
            for i, name in enumerate(label_names):
                name_lower = name.lower()
                if 'input' in name_lower or 'low' in name_lower:
                    low_label_index = i
                elif 'gt' in name_lower or 'high' in name_lower:
                    high_label_index = i
            
            if low_label_index == -1 or high_label_index == -1:
                raise Exception(f"Could not find 'low'/'high' or 'Input'/'GT' in label names: {label_names}")

            print(f"Auto-detected labels: LOW='{label_names[low_label_index]}' (Index={low_label_index}), HIGH='{label_names[high_label_index]}' (Index={high_label_index})")
            
        except Exception as e:
            print(f"Fatal Error: Could not determine labels. {e}")
            continue # Skip to next split

        # Create output directories (this logic is unchanged)
        low_dir = os.path.join(args.output_dir, dataset_name, split_name, "low")
        high_dir = os.path.join(args.output_dir, dataset_name, split_name, "high")
        mask_dir = os.path.join(args.output_dir, dataset_name, split_name, "masks")

        os.makedirs(low_dir, exist_ok=True)
        os.makedirs(high_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        
        print(f"Saving LOW images to: {low_dir}")
        print(f"Saving HIGH images to: {high_dir}")
        print(f"Saving MASKS to: {mask_dir}")

        low_saved_count = 0
        high_saved_count = 0
        unmatched_labels_found = set()

        # This loop logic is unchanged
        for i, item in enumerate(tqdm(split_data, desc=f"Processing {dataset_name} {split_name} split")):
            try:
                input_image = item['image'].convert("RGB") 
                label = item['label'] # This will be 0 or 1
                output_filename = f"{split_name}_{i:05d}.png"

                if label == low_label_index:
                    low_path = os.path.join(low_dir, output_filename)
                    input_image.save(low_path)
                    low_saved_count += 1
                    
                    mask_path = os.path.join(mask_dir, output_filename)
                    run_automatic_segmentation(model, input_image, mask_path, args)
                    
                elif label == high_label_index:
                    high_path = os.path.join(high_dir, output_filename)
                    input_image.save(high_path)
                    high_saved_count += 1
                
                else:
                    unmatched_labels_found.add(label)

            except Exception as e:
                print(f"\nWarning: Failed to process item {i}")

if __name__ == "__main__":
    # Add parent directory to Python path to find 'fastsam'
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    args = parse_args()
    main(args)