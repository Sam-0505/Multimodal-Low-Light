import os
import argparse
from PIL import Image
import numpy as np
from transformers import pipeline

parser = argparse.ArgumentParser("depth-anything")
parser.add_argument('--source_dir', type=str, default='data/LOL/test15/low', help='directory of images to process')
args = parser.parse_args()

# Output directory: change 'low' to 'low_depth' (works for both test and train folders)
depth_dir = args.source_dir.replace("low", "low_depth")
os.makedirs(depth_dir, exist_ok=True)

# Load Depth-Anything pipeline
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

for i, filename in enumerate(os.listdir(args.source_dir)):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        print(f'{i}th image: {filename}')
        img_path = os.path.join(args.source_dir, filename)
        image = Image.open(img_path)
        output = pipe(image)
        depth_pil = output["depth"]  # PIL Image (high dynamic range, single-channel)
        depth = np.array(depth_pil).astype(np.float32)  # (H, W)
        # Scale to 0-65535 for 16-bit PNG
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_16bit = (depth_norm * 65535).astype(np.uint16)
        save_path = os.path.join(depth_dir, f"{os.path.splitext(filename)[0]}_depth.png")
        Image.fromarray(depth_16bit).save(save_path)
        print(f"Saved depth to {save_path}")
