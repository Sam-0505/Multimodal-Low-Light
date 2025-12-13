import argparse
import asyncio
import logging
import cv2
import numpy as np
import torch
import sys
import os
import time
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder
import uuid
from collections import deque
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torch.autograd import Variable

# Add parent directory: Multimodal-Low-Light
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

# Add SCI model path
SCI_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "SCI-main", "SCI-main"))
sys.path.append(SCI_DIR)

from inference import WakeupDarknessEngine
from model import Network_woCalibrate
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from transformers import pipeline

# Import SCI model
try:
    from CVPR.model import Finetunemodel
    SCI_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import SCI model: {e}")
    SCI_AVAILABLE = False

# Initialize WakeupDarkness Engine
engine = WakeupDarknessEngine(
    enhance_weights=r"C:\Users\samir\Downloads\Code_Projects\Multimodal-Low-Light\weights\final\my_model.pt",
    fastsam_weights=r"C:\Users\samir\Downloads\Code_Projects\Multimodal-Low-Light\weights\FastSAM-s.pt"
)

# Initialize Second Model (Network_woCalibrate with SAM and Depth)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load EnhanceNetwork
enhance_model = Network_woCalibrate()
weights_dict = torch.load(r"C:\Users\samir\Downloads\Code_Projects\Multimodal-Low-Light\weights\final\wakeupdarkness.pt", map_location=device)
model_dict = enhance_model.state_dict()
weights_dict = {k: v for k, v in weights_dict.items() if k in model_dict}
model_dict.update(weights_dict)
enhance_model.load_state_dict(model_dict)
enhance_model.to(device)
enhance_model.eval()

# Load SAM
sam = sam_model_registry["vit_h"](checkpoint=r"..\segment_anything\sam_vit_h_4b8939.pth")
sam.to(device=device)
sam.eval()
mask_generator = SamAutomaticMaskGenerator(sam)

# Load Depth-Anything
depth_pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-large-hf", device=device)

# Load SCI Model
if SCI_AVAILABLE:
    sci_model = Finetunemodel(os.path.join(SCI_DIR, "CVPR", "weights", "difficult.pt"))
    sci_model = sci_model.cuda()
    sci_model.eval()
    print("SCI model loaded successfully!")
else:
    sci_model = None
    print("SCI model not available")

print("All available models loaded successfully!")

# Store active prompts for each session (Session ID -> Prompt String)
active_prompts = {}

def is_low_light(img_bgr, threshold=60):
    """Check if image is low-light based on average brightness."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness < threshold, brightness

def draw_info(img, fps, is_enhanced, brightness, model_name):
    """Draw FPS, brightness, enhancement status, and model name on frame."""
    text1 = f"{model_name}"
    text2 = f"FPS: {fps:.1f}"
    text3 = f"Brightness: {brightness:.1f}"
    text4 = "Enhanced: YES" if is_enhanced else "Enhanced: NO"

    cv2.putText(img, text1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(img, text2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(img, text3, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(img, text4, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return img

def process_sem_input_sam(mask_generator, image_pil, device):
    """Process SAM masks for the second model."""
    image_np_rgb = np.array(image_pil)
    masks = mask_generator.generate(image_np_rgb)
    
    if not masks:
        H, W, _ = image_np_rgb.shape
        return torch.zeros(1, 3, H, W, device=device, dtype=torch.float)

    H, W = masks[0]['segmentation'].shape
    merged_mask_np = np.zeros((H, W), dtype=bool)
    for mask_dict in masks:
        merged_mask_np |= mask_dict['segmentation']
        
    merged_mask_tensor = torch.from_numpy(merged_mask_np).float().to(device)
    sem_tensor = merged_mask_tensor.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
    
    return sem_tensor

def process_depth_input(depth_pipe, image_pil, device):
    """Process depth map for the second model."""
    output = depth_pipe(image_pil)
    depth_pil = output["depth"]
    depth_pil_rgb = depth_pil.convert("RGB")
    depth_tensor = to_tensor(depth_pil_rgb).unsqueeze(0).to(device)
    return depth_tensor


class VideoTransformTrack(MediaStreamTrack):
    """Base class for video transformation with FPS tracking."""
    kind = "video"

    def __init__(self, track, session_id, model_name, threshold=100):
        super().__init__()
        self.track = track
        self.session_id = session_id
        self.model_name = model_name
        self.threshold = threshold

        # Shared states
        self.latest_raw_frame = None
        self.latest_enhanced_frame = None
        self.enhancing = False
        self.frame_hash = None  # Track if frame has changed
        
        # FPS tracking for output frames
        self.output_frame_times = deque(maxlen=30)
        self.avg_fps = 0.0
        
        # Enhancement tracking
        self.is_currently_enhanced = False
        self.current_brightness = 0.0
        self.last_output_time = None

        # Worker loop
        self.loop = asyncio.get_event_loop()
        self.loop.create_task(self.enhancement_worker())

    async def enhancement_worker(self):
        """Override this in subclasses."""
        raise NotImplementedError

    async def recv(self):
        frame = await self.track.recv()

        # Convert raw input to numpy
        img_bgr = frame.to_ndarray(format="bgr24")

        # Update "latest raw frame"
        self.latest_raw_frame = img_bgr
        
        # Always update brightness from current frame
        _, brightness = is_low_light(img_bgr, self.threshold)
        self.current_brightness = brightness

        # Choose output frame
        if self.latest_enhanced_frame is not None and self.is_currently_enhanced:
            output = self.latest_enhanced_frame.copy()
        else:
            # Use original frame without copying to preserve quality
            output = img_bgr
            
            # For non-enhanced frames, track FPS of raw passthrough
            current_time = time.time()
            self.output_frame_times.append(current_time)
            
            if len(self.output_frame_times) > 1:
                time_diff = self.output_frame_times[-1] - self.output_frame_times[0]
                if time_diff > 0:
                    self.avg_fps = (len(self.output_frame_times) - 1) / time_diff
        
        # Draw info on the frame (make a copy first to avoid modifying original)
        output_with_info = output.copy()
        output_with_info = draw_info(output_with_info, self.avg_fps, self.is_currently_enhanced, 
                          self.current_brightness, self.model_name)

        new_frame = frame.from_ndarray(output_with_info, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base

        return new_frame


class WakeupDarknessTrack(VideoTransformTrack):
    """Video track using WakeupDarknessEngine."""
    
    def __init__(self, track, session_id, threshold=100):
        super().__init__(track, session_id, "Our Model", threshold)

    async def enhancement_worker(self):
        """Background worker for WakeupDarknessEngine."""
        while True:
            await asyncio.sleep(0.01)  # Reduced frequency

            if self.latest_raw_frame is None:
                await asyncio.sleep(0.005)
                continue

            if self.enhancing:
                await asyncio.sleep(0.001)
                continue

            # Check if we have a new frame
            current_hash = hash(self.latest_raw_frame.tobytes())
            if current_hash == self.frame_hash:
                # Same frame, skip processing
                await asyncio.sleep(0.05)
                continue
            
            self.frame_hash = current_hash
            raw_frame = self.latest_raw_frame.copy()
            
            # Check if enhancement is needed
            is_low, brightness = is_low_light(raw_frame, self.threshold)
            
            if not is_low:
                self.latest_enhanced_frame = None
                self.is_currently_enhanced = False
                await asyncio.sleep(0.01)
                continue
            
            self.enhancing = True

            def run_inference():
                return engine.enhance_image(raw_frame)

            try:
                enhanced = await asyncio.get_event_loop().run_in_executor(
                    None, run_inference
                )
                self.latest_enhanced_frame = enhanced
                self.is_currently_enhanced = True
                
                # Track FPS
                current_time = time.time()
                self.output_frame_times.append(current_time)
                
                if len(self.output_frame_times) > 1:
                    time_diff = self.output_frame_times[-1] - self.output_frame_times[0]
                    if time_diff > 0:
                        self.avg_fps = (len(self.output_frame_times) - 1) / time_diff
                        
            except Exception as e:
                print("WakeupDarkness enhancement failed:", e)
                import traceback
                traceback.print_exc()
                self.is_currently_enhanced = False

            self.enhancing = False


class NetworkWoCalibrateTrack(VideoTransformTrack):
    """Video track using Network_woCalibrate with SAM and Depth."""
    
    def __init__(self, track, session_id, threshold=100):
        super().__init__(track, session_id, "Network+SAM+Depth", threshold)

    async def enhancement_worker(self):
        """Background worker for Network_woCalibrate."""
        while True:
            await asyncio.sleep(0.01)  # Reduced frequency

            if self.latest_raw_frame is None:
                await asyncio.sleep(0.005)
                continue

            if self.enhancing:
                await asyncio.sleep(0.001)
                continue

            # Check if we have a new frame
            current_hash = hash(self.latest_raw_frame.tobytes())
            if current_hash == self.frame_hash:
                # Same frame, skip processing
                await asyncio.sleep(0.05)
                continue
            
            self.frame_hash = current_hash
            raw_frame = self.latest_raw_frame.copy()
            
            # Check if enhancement is needed
            is_low, brightness = is_low_light(raw_frame, self.threshold)
            
            if not is_low:
                self.latest_enhanced_frame = None
                self.is_currently_enhanced = False
                await asyncio.sleep(0.01)
                continue
            
            self.enhancing = True

            def run_inference():
                # Convert BGR to RGB PIL Image
                image_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_rgb)
                
                # Process inputs
                with torch.no_grad():
                    in_tensor = to_tensor(image_pil).unsqueeze(0).to(device)
                    sem_tensor = process_sem_input_sam(mask_generator, image_pil, device)
                    depth_tensor = process_depth_input(depth_pipe, image_pil, device)
                    
                    # Run model
                    i, r, d = enhance_model(in_tensor, sem_tensor, depth_tensor)
                    
                    # Convert output tensor to numpy BGR
                    output_np = r.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                    output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
                    output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
                    
                return output_bgr

            try:
                enhanced = await asyncio.get_event_loop().run_in_executor(
                    None, run_inference
                )
                self.latest_enhanced_frame = enhanced
                self.is_currently_enhanced = True
                
                # Track FPS
                current_time = time.time()
                self.output_frame_times.append(current_time)
                
                if len(self.output_frame_times) > 1:
                    time_diff = self.output_frame_times[-1] - self.output_frame_times[0]
                    if time_diff > 0:
                        self.avg_fps = (len(self.output_frame_times) - 1) / time_diff
                        
            except Exception as e:
                print("Network_woCalibrate enhancement failed:", e)
                import traceback
                traceback.print_exc()
                self.is_currently_enhanced = False

            self.enhancing = False


class SCITrack(VideoTransformTrack):
    """Video track using SCI model."""
    
    def __init__(self, track, session_id, threshold=100):
        super().__init__(track, session_id, "SCI Model", threshold)

    async def enhancement_worker(self):
        """Background worker for SCI model."""
        if not SCI_AVAILABLE or sci_model is None:
            print("SCI model not available")
            return
        
        print("SCI enhancement worker started")
            
        while True:
            await asyncio.sleep(0.01)  # Reduced frequency

            if self.latest_raw_frame is None:
                await asyncio.sleep(0.005)
                continue

            if self.enhancing:
                await asyncio.sleep(0.001)
                continue

            # Check if we have a new frame
            current_hash = hash(self.latest_raw_frame.tobytes())
            if current_hash == self.frame_hash:
                # Same frame, skip processing
                await asyncio.sleep(0.05)
                continue
            
            self.frame_hash = current_hash
            raw_frame = self.latest_raw_frame.copy()
            
            # Check if enhancement is needed
            is_low, brightness = is_low_light(raw_frame, self.threshold)
            
            if not is_low:
                self.latest_enhanced_frame = None
                self.is_currently_enhanced = False
                await asyncio.sleep(0.01)
                continue
            
            print(f"SCI model - Starting enhancement (brightness: {brightness:.1f})")
            self.enhancing = True

            def run_inference():
                # Convert BGR to RGB PIL Image
                image_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_rgb)
                
                # Convert to tensor and normalize
                with torch.no_grad():
                    # SCI expects input in [0, 1] range
                    input_tensor = to_tensor(image_pil).unsqueeze(0).cuda()
                    print(f"SCI input tensor shape: {input_tensor.shape}, range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
                    
                    # Run SCI model
                    i, r = sci_model(input_tensor)
                    print(f"SCI output tensor shape: {r.shape}, range: [{r.min():.3f}, {r.max():.3f}]")
                    
                    # Convert output tensor to numpy BGR
                    output_np = r.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                    output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
                    output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
                    
                return output_bgr

            try:
                enhanced = await asyncio.get_event_loop().run_in_executor(
                    None, run_inference
                )
                self.latest_enhanced_frame = enhanced
                self.is_currently_enhanced = True
                print(f"SCI model - Enhancement complete, output shape: {enhanced.shape}")
                
                # Track FPS
                current_time = time.time()
                self.output_frame_times.append(current_time)
                
                if len(self.output_frame_times) > 1:
                    time_diff = self.output_frame_times[-1] - self.output_frame_times[0]
                    if time_diff > 0:
                        self.avg_fps = (len(self.output_frame_times) - 1) / time_diff
                        
            except Exception as e:
                print("SCI enhancement failed:", e)
                import traceback
                traceback.print_exc()
                self.is_currently_enhanced = False

            self.enhancing = False


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Helper to manage RTCPeerConnections
pcs = set()

@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    session_id = params.get("session_id")
    model_type = params.get("model_type", "wakeup")  # "wakeup", "network", or "sci"
    threshold = params.get("threshold", 100)  # Default threshold

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    # Handle the incoming video track
    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            print(f"Video track received for session {session_id}, model: {model_type}, threshold: {threshold}")
            
            # Choose which model track to use
            if model_type == "wakeup":
                local_video = WakeupDarknessTrack(track, session_id, threshold)
            elif model_type == "network":
                local_video = NetworkWoCalibrateTrack(track, session_id, threshold)
            elif model_type == "sci":
                local_video = SCITrack(track, session_id, threshold)
            else:
                print(f"Unknown model type: {model_type}")
                return
            
            pc.addTrack(local_video)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }

@app.post("/update_prompt")
async def update_prompt(request: Request):
    data = await request.json()
    session_id = data.get("session_id")
    prompt = data.get("prompt")
    active_prompts[session_id] = prompt
    return {"status": "ok", "prompt": prompt}

@app.on_event("shutdown")
async def on_shutdown():
    # Close all connections on shutdown
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)