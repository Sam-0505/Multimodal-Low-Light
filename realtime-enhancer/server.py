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

def is_low_light(img_bgr, threshold=60):
    # Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Mean brightness
    brightness = np.mean(gray)
    return brightness < threshold, brightness

def draw_info(img, fps, is_low, brightness):
    text1 = f"FPS: {fps:.1f}"
    text2 = f"Brightness: {brightness:.1f}"
    text3 = "Enhanced: YES" if is_low else "Enhanced: NO"

    cv2.putText(img, text1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(img, text2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(img, text3, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    return img

import asyncio
from aiortc import MediaStreamTrack

class VideoTransformTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track, session_id):
        super().__init__()
        self.track = track
        self.session_id = session_id

        # Shared states
        self.latest_raw_frame = None        # newest raw frame (ndarray)
        self.latest_enhanced_frame = None   # last enhanced output
        self.enhancing = False              # is model running?

        # Worker loop
        self.loop = asyncio.get_event_loop()
        self.loop.create_task(self.enhancement_worker())

    async def enhancement_worker(self):
        """Background worker that enhances newest frame asynchronously."""
        while True:
            await asyncio.sleep(0)  # let event loop breathe

            # No new raw frame available
            if self.latest_raw_frame is None:
                await asyncio.sleep(0.005)
                continue

            # Enhancement already in progress
            if self.enhancing:
                await asyncio.sleep(0.001)
                continue

            # Take latest raw frame and launch enhancement
            raw_frame = self.latest_raw_frame.copy()
            self.enhancing = True

            def run_inference():
                return engine.enhance_image(raw_frame)

            try:
                enhanced = await asyncio.get_event_loop().run_in_executor(
                    None, run_inference
                )
                self.latest_enhanced_frame = enhanced
            except Exception as e:
                print("Enhancement failed:", e)

            self.enhancing = False

    async def recv(self):
        frame = await self.track.recv()

        # Convert raw input to numpy
        img_bgr = frame.to_ndarray(format="bgr24")

        # Update "latest raw frame" (drop older ones)
        self.latest_raw_frame = img_bgr

        # Choose output frame: enhanced or fallback to raw
        output = (
            self.latest_enhanced_frame
            if self.latest_enhanced_frame is not None
            else img_bgr
        )

        new_frame = frame.from_ndarray(output, format="bgr24")
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