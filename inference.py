import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor

# --- Path Setup ---
# Ensure FastSAM and current dir are in path
FASTSAM_DIR = os.path.join(os.path.dirname(__file__), 'FastSAM')
sys.path.insert(0, FASTSAM_DIR)
sys.path.insert(0, os.path.dirname(__file__))

# --- Model Imports ---
from model import Network_woCalibrate
from FastSAM.fastsam import FastSAM, FastSAMPrompt
from transformers import pipeline

class WakeupDarknessEngine:
    def __init__(self, 
                 enhance_weights="weights/final/weights_999.pt",
                 fastsam_weights="weights/FastSAM-s.pt",
                 depth_model_name="depth-anything/Depth-Anything-V2-Small-hf",
                 gpu_id=0):
        
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        print(f"Initializing Wakeup-Darkness Engine on {self.device}...")

        # 1. Load EnhanceNetwork
        self.enhance_model = Network_woCalibrate()
        weights_dict = torch.load(enhance_weights, map_location=self.device)
        model_dict = self.enhance_model.state_dict()
        # Filter and load weights
        weights_dict = {k: v for k, v in weights_dict.items() if k in model_dict}
        model_dict.update(weights_dict)
        self.enhance_model.load_state_dict(model_dict)
        self.enhance_model.to(self.device)
        self.enhance_model.eval()
        print("  - EnhanceNetwork loaded.")

        # 2. Load FastSAM
        self.fastsam_model = FastSAM(fastsam_weights)
        self.fastsam_model.to(self.device)
        print("  - FastSAM loaded.")

        # 3. Load Depth-Anything
        self.depth_pipe = pipeline(task="depth-estimation", model=depth_model_name, device=gpu_id)
        print("  - Depth-Anything loaded.")

        # Warmup
        self._warmup()
        print("Engine Ready.")

    def _warmup(self):
        """Runs a dummy pass to initialize CUDA context."""
        with torch.no_grad():
            dummy = Image.new('RGB', (640, 480))
            self.enhance_image(dummy)

    def _process_sem_input(self, image_pil):
        """Internal helper to run FastSAM and generate semantic tensor."""
        everything_results = self.fastsam_model(
            image_pil,
            device=self.device,
            retina_masks=True,
            imgsz=1024,
            conf=0.4,
            iou=0.9,
            verbose=False # Silence FastSAM output
        )
        
        prompt_process = FastSAMPrompt(image_pil, everything_results, device=self.device)
        ann = prompt_process.everything_prompt()
        
        ann_tensor = None

        # Robust Check (From your script)
        if isinstance(ann, torch.Tensor):
            if ann.numel() != 0:
                ann_tensor = ann
        elif isinstance(ann, list):
            if ann:
                try:
                    ann_tensor = torch.stack(ann)
                except:
                    pass

        if ann_tensor is None or ann_tensor.numel() == 0:
            H, W = image_pil.height, image_pil.width
            return torch.zeros(1, 3, H, W, device=self.device, dtype=torch.float)

        sem_mask, _ = torch.max(ann_tensor, dim=0)
        sem_tensor = sem_mask.float().unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
        return sem_tensor

    def _process_depth_input(self, image_pil):
        """Internal helper to run Depth-Anything."""
        output = self.depth_pipe(image_pil)
        depth_pil = output["depth"].convert("RGB")
        depth_tensor = to_tensor(depth_pil).unsqueeze(0).to(self.device)
        return depth_tensor

    def enhance_image(self, image_input):
        """
        Main entry point.
        Args:
            image_input: Can be a PIL Image or a Numpy Array (OpenCV BGR format).
        Returns:
            Numpy Array (BGR format) ready for OpenCV/Video writing.
        """
        # 1. Standardize Input to PIL RGB
        if isinstance(image_input, np.ndarray):
            # OpenCV (BGR) -> PIL (RGB)
            image_pil = Image.fromarray(image_input[:, :, ::-1]) 
        else:
            image_pil = image_input

        with torch.no_grad():
            # 2. Prepare Inputs
            in_tensor = to_tensor(image_pil).unsqueeze(0).to(self.device)
            sem_tensor = self._process_sem_input(image_pil)
            depth_tensor = self._process_depth_input(image_pil)

            # 3. Run Inference
            # Based on your script, model returns a tuple (i, r, d). 
            # Assuming 'i' or the first element is the final enhanced image.
            output_tuple = self.enhance_model(in_tensor, sem_tensor, depth_tensor)
            
            # Handle return type based on your model.py structure
            if isinstance(output_tuple, tuple):
                enhanced_tensor = output_tuple[1] # Usually the first item is the result
            else:
                enhanced_tensor = output_tuple

            # 4. Post-process (Tensor -> Numpy BGR)
            enhanced_tensor = enhanced_tensor.squeeze(0).clamp(0, 1).cpu()
            enhanced_np_rgb = (enhanced_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            enhanced_np_bgr = enhanced_np_rgb[:, :, ::-1]
            
            return enhanced_np_bgr