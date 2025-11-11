import os
import re
import glob
import argparse  # <-- Added import

def rename_files(gt_dir, offset):  # <-- Now accepts arguments
    """
    Finds all .png and .jpg files in the gt_dir, extracts the trailing
    number from their filename, adds the offset, and renames the file
    while preserving zero-padding.
    """
    
    # Check if the directory exists
    if not os.path.isdir(gt_dir):
        print(f"Error: Directory '{gt_dir}' does not exist.")
        return

    print(f"Target directory: {gt_dir}")
    print(f"Adding offset: {offset}")
    print("-" * 30)

    # Get all png and jpg files by combining two glob searches
    png_files = glob.glob(os.path.join(gt_dir, '*.png'))
    jpg_files = glob.glob(os.path.join(gt_dir, '*.jpg'))
    all_files = png_files + jpg_files

    if not all_files:
        print("No .png or .jpg files found to rename.")
        return

    renamed_count = 0
    skipped_count = 0
    
    for old_path in all_files:
        old_name = os.path.basename(old_path)
        
        # Split filename into basename (e.g., "train_00038") and extension (e.g., ".png")
        basename, extension = os.path.splitext(old_name)

        # Regex to find the last sequence of digits at the end of the basename
        match = re.search(r'(\d+)$', basename)

        if match:
            number_str = match.group(1)
            # Get the part before the number (e.g., "train_")
            prefix = basename[:match.start()]
            
            # Get the length of the original number string to preserve zero-padding
            number_len = len(number_str)
            
            try:
                # Convert number to integer and add the offset
                number_int = int(number_str) + offset
                
                # Format the new number back into a string with the original zero-padding
                new_number_str = str(number_int).zfill(number_len)
                
                # Construct the new filename and full path
                new_name = f"{prefix}{new_number_str}{extension}"
                new_path = os.path.join(gt_dir, new_name)
                
                # Rename the file
                if old_path != new_path:
                    print(f"Renaming: {old_name}  ->  {new_name}")
                    os.rename(old_path, new_path)
                    renamed_count += 1
                else:
                    print(f"Skipping (no change): {old_name}")
                    skipped_count += 1
                        
            except ValueError:
                print(f"Warning: Could not parse number {number_str} in '{old_name}'. Skipping.")
                skipped_count += 1
        else:
            print(f"Warning: Could not parse number in '{old_name}'. Skipping.")
            skipped_count += 1

    print("-" * 30)
    print("Renaming complete.")
    print(f"Files renamed: {renamed_count}")
    print(f"Files skipped: {skipped_count}")

# This makes the script runnable when you call `python rename_gt.py`
if __name__ == "__main__":
    # --- Argument parsing logic ---
    parser = argparse.ArgumentParser(description="Rename GT files with a numerical offset.")
    
    parser.add_argument('--gt_dir', 
                        type=str, 
                        required=True, 
                        help='Path to the ground truth directory.')
    
    parser.add_argument('--offset', 
                        type=int, 
                        default=100, 
                        help='Numerical offset to add to filenames (default: 100).')
    
    args = parser.parse_args()
    
    # Call the function with the parsed arguments
    rename_files(args.gt_dir, args.offset)