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

# Initialize Engine
engine = WakeupDarknessEngine(
    enhance_weights=r"C:\Users\samir\Downloads\Code_Projects\Multimodal-Low-Light\weights\final\weights_999.pt",
    fastsam_weights=r"C:\Users\samir\Downloads\Code_Projects\Multimodal-Low-Light\weights\FastSAM-s.pt"
)

# Store active prompts for each session (Session ID -> Prompt String)
active_prompts = {}

class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from another track.
    """
    kind = "video"

    def __init__(self, track, session_id):
        super().__init__()
        self.track = track
        self.session_id = session_id
        
        # --- NEW: Frame Skipping State ---
        self.frame_count = 0
        self.last_enhanced_frame = None
        self.skip_interval = 5  # Enhance every 5th frame

    async def recv(self):
        # 1. Get the next frame from the input stream
        frame = await self.track.recv()
        
        # Increment counter
        self.frame_count += 1
        
        # 2. DECISION: Enhance or Reuse?
        if self.frame_count % self.skip_interval == 0:
            # --- Path A: Run Heavy Inference ---
            
            # Get numpy array (OpenCV format)
            img_bgr = frame.to_ndarray(format="bgr24")
            
            # Run the engine (This takes ~100ms)
            # You can also pass the prompt here if you modify your engine to accept it
            # prompt = active_prompts.get(self.session_id, "")
            try:
                enhanced_img_bgr = engine.enhance_image(img_bgr)
                
                # Update the "Cache"
                self.last_enhanced_frame = enhanced_img_bgr
            except Exception as e:
                print(f"Inference failed: {e}")
                # Fallback to original if model fails
                self.last_enhanced_frame = img_bgr
                
        else:
            # --- Path B: Skip Inference ---
            # If we haven't enhanced a frame yet (first 4 frames), use original
            if self.last_enhanced_frame is None:
                img_bgr = frame.to_ndarray(format="bgr24")
                self.last_enhanced_frame = img_bgr
            
            # Use the cached enhanced frame
            enhanced_img_bgr = self.last_enhanced_frame

        # 3. Rebuild the video frame
        # We must use the timestamps from the *current* incoming frame 'frame'
        # so the browser plays it in sync, even if the image data is old.
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