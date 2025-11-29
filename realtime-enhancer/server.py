# server.py
import argparse
import asyncio
import logging
import cv2
import numpy as np
import torch
import sys
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder
import uuid

# Add parent directory: Multimodal-Low-Light
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

from inference import WakeupDarknessEngine


engine = WakeupDarknessEngine(
    enhance_weights=r"C:\Users\samir\Downloads\Code_Projects\Multimodal-Low-Light\weights\final\weights_999.pt",
    fastsam_weights=r"C:\Users\samir\Downloads\Code_Projects\Multimodal-Low-Light\weights\FastSAM-s.pt"
)
# --- 1. MOCK MODEL LOADING (Replace with actual Wakeup-Darkness code) ---
class WakeupDarknessModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Model loaded on {self.device}")
        # self.model = load_your_actual_model() 

    def predict(self, frame_numpy, prompt_text=""):
        # This is where your paper's inference logic goes
        # frame_numpy is (H, W, 3) BGR uint8
        
        # Simulating processing: Draw a box based on prompt length
        # In reality: return self.model(frame_numpy, prompt)
        
        # Fake "Enhancement": brightening the image slightly
        enhanced = cv2.convertScaleAbs(frame_numpy, alpha=1.2, beta=10)
        
        # Visualizing the prompt on screen for debug
        cv2.putText(enhanced, f"Enhancing: {prompt_text}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return enhanced

model = WakeupDarknessModel()
# ----------------------------------------------------------------------

# Store active prompts for each session (Session ID -> Prompt String)
active_prompts = {}

class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """
    kind = "video"

    def __init__(self, track, session_id):
        super().__init__()
        self.track = track
        self.session_id = session_id

    async def recv(self):
        frame = await self.track.recv()
        
        # 1. Get numpy array (OpenCV format)
        img_bgr = frame.to_ndarray(format="bgr24")
        
        # 2. CALL YOUR NEW FUNCTION
        # The engine handles all the PIL conversion, SAM, Depth, and Enhancement internally
        enhanced_img_bgr = engine.enhance_image(img_bgr)
        
        # 3. Return frame
        new_frame = frame.from_ndarray(enhanced_img_bgr, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Helper to manage RTCPeerConnections
pcs = set()

@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    session_id = params.get("session_id")

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
            print(f"Video track received for session {session_id}")
            # Wrap the input track with our Enhancement Processor
            local_video = VideoTransformTrack(track, session_id)
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
    # SSL is usually required for webcam access on non-localhost
    uvicorn.run(app, host="0.0.0.0", port=8000)