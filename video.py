import os
import sys
import cv2
import torch
import numpy as np
import argparse
import time
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor

# --- IMPORTS FROM YOUR PROJECT ---
try:
    from model import Network_woCalibrate
    from transformers import pipeline
    
    # FastSAM path fix
    sys.path.append(os.path.join(os.path.dirname(__file__), 'FastSAM'))
    from FastSAM.fast_sam import FastSAM, FastSAMPrompt
except ImportError as e:
    print(f"Import Error: {e}. Ensure you are running this from your project root.")

class FeaturePropagator:
    """
    Handles the 'Keyframe + Optical Flow' logic to speed up video inference.
    """
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.keyframe_interval = args.interval
        
        # --- Load Models ---
        print("Loading models...")
        # 1. Enhance Network (Lightweight - Runs every frame)
        self.enhance_model = Network_woCalibrate().to(device)
        self.load_weights(self.enhance_model, args.enhance_weights)
        self.enhance_model.eval()
        
        # 2. FastSAM (Heavy - Runs only on Keyframes)
        self.fastsam = FastSAM(args.fastsam_weights)
        
        # 3. Depth Anything (Heavy - Runs only on Keyframes)
        self.depth_pipe = pipeline(task="depth-estimation", model=args.depth_model_name, device=device)
        
        # 4. Optical Flow (OpenCV DIS - Very Fast)
        self.dis_flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        
        # State storage for propagation
        self.last_keyframe_gray = None 
        self.last_sem_feat = None       
        self.last_depth_feat = None     
        self.frame_count = 0

    def load_weights(self, model, path):
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    def get_fastsam_feat(self, image_pil):
        """Run FastSAM and return (1, 3, H, W) tensor"""
        results = self.fastsam(image_pil, device=self.device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
        prompt_process = FastSAMPrompt(image_pil, results, device=self.device)
        ann = prompt_process.everything_prompt()
        
        if not ann: 
             return torch.zeros(1, 3, image_pil.height, image_pil.width, device=self.device)
             
        if isinstance(ann, list):
            try: ann = torch.stack(ann)
            except: return torch.zeros(1, 3, image_pil.height, image_pil.width, device=self.device)
            
        if ann.numel() == 0:
             return torch.zeros(1, 3, image_pil.height, image_pil.width, device=self.device)

        mask, _ = torch.max(ann, dim=0)
        return mask.float().unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)

    def get_depth_feat(self, image_pil):
        """Run DepthAnything and return (1, 3, H, W) tensor"""
        out = self.depth_pipe(image_pil)["depth"]
        return to_tensor(out.convert("RGB")).unsqueeze(0).to(self.device)

    def warp_feature(self, feature, flow):
        """Warp a feature map using Optical Flow."""
        B, C, H, W = feature.size()
        grid_y, grid_x = torch.meshgrid(torch.arange(0, H, device=self.device), torch.arange(0, W, device=self.device), indexing='ij')
        flow_tensor = torch.from_numpy(flow).to(self.device)
        
        vgrid_x = grid_x - flow_tensor[:,:,0]
        vgrid_y = grid_y - flow_tensor[:,:,1]
        
        vgrid_x = 2.0 * vgrid_x / (W - 1) - 1.0
        vgrid_y = 2.0 * vgrid_y / (H - 1) - 1.0
        
        vgrid = torch.stack((vgrid_x, vgrid_y), dim=2).unsqueeze(0)
        warped_feature = F.grid_sample(feature, vgrid, mode='bilinear', padding_mode='border', align_corners=True)
        return warped_feature

    def process_video(self):
        cap = cv2.VideoCapture(self.args.input_video)
        fps_in = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.args.output_video, fourcc, fps_in, (width, height))
        
        print(f"Processing {self.args.input_video} ({total_frames} frames)")
        print(f"Keyframe Interval: {self.keyframe_interval}")

        # --- TIMING START ---
        start_time = time.time()
        
        pbar = tqdm(total=total_frames, unit="frame")
        
        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            image_pil = Image.fromarray(frame_rgb)
            
            input_tensor = to_tensor(image_pil).unsqueeze(0).to(self.device)
            
            is_keyframe = (self.frame_count % self.keyframe_interval == 0)
            
            if is_keyframe:
                # Heavy Path
                self.last_sem_feat = self.get_fastsam_feat(image_pil)
                self.last_depth_feat = self.get_depth_feat(image_pil)
                self.last_keyframe_gray = frame_gray
                
                sem_input = self.last_sem_feat
                depth_input = self.last_depth_feat
                pbar.set_postfix_str("Mode: Keyframe (Slow)")
            else:
                # Light Path (Warping)
                if self.last_keyframe_gray is not None:
                    flow = self.dis_flow.calc(self.last_keyframe_gray, frame_gray, None)
                    sem_input = self.warp_feature(self.last_sem_feat, flow)
                    depth_input = self.warp_feature(self.last_depth_feat, flow)
                    
                    self.last_keyframe_gray = frame_gray 
                    self.last_sem_feat = sem_input
                    self.last_depth_feat = depth_input
                    pbar.set_postfix_str("Mode: Warping (Fast)")
                else:
                    sem_input = self.last_sem_feat
                    depth_input = self.last_depth_feat

            with torch.no_grad():
                _, enhanced_tensor, _ = self.enhance_model(input_tensor, sem_input, depth_input)

            enhanced_np = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            enhanced_np = np.clip(enhanced_np * 255, 0, 255).astype(np.uint8)
            enhanced_bgr = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2BGR)
            out.write(enhanced_bgr)
            
            self.frame_count += 1
            pbar.update(1)

        # --- TIMING END ---
        end_time = time.time()
        total_time = end_time - start_time
        avg_fps = self.frame_count / total_time if total_time > 0 else 0

        cap.release()
        out.release()
        pbar.close()
        
        print("\n" + "="*30)
        print(f"PROCESSING COMPLETE")
        print(f"Total Time:  {total_time:.2f} seconds")
        print(f"Total Frames: {self.frame_count}")
        print(f"Average FPS: {avg_fps:.2f}")
        print("="*30)
        print(f"Video saved to: {self.args.output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output_video', type=str, default='output.mp4', help='Path to output video')
    parser.add_argument('--interval', type=int, default=10, help='Keyframe interval')
    parser.add_argument('--enhance_weights', type=str, default="weights/pretrained_SCI/medium.pt")
    parser.add_argument('--fastsam_weights', type=str, default="weights/FastSAM-s.pt")
    parser.add_argument('--depth_model_name', type=str, default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    propagator = FeaturePropagator(args, device)
    propagator.process_video()