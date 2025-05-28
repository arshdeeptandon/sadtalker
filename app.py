from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import asyncio
from typing import Optional
import shutil
import torch
from src.gradio_demo import SadTalker
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import logging
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.utils.init_path import init_path

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for preloaded models
sad_talker = None
preprocess_model = None
audio_to_coeff = None
animate_from_coeff = None
sadtalker_paths = None

def enable_fp16():
    """Enable fp16 for faster inference"""
    if torch.cuda.is_available():
        # Enable automatic mixed precision
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        return True
    return False

@app.on_event("startup")
async def startup_event():
    """Initialize all models during server startup"""
    global sad_talker, preprocess_model, audio_to_coeff, animate_from_coeff, sadtalker_paths
    
    try:
        logger.info("Starting model initialization...")
        
        # Enable fp16 if available
        fp16_enabled = enable_fp16()
        logger.info(f"FP16 enabled: {fp16_enabled}")
        
        # Initialize paths
        current_root_path = os.path.dirname(os.path.abspath(__file__))
        sadtalker_paths = init_path('checkpoints', os.path.join(current_root_path, 'src/config'), 256, False, 'full')
        logger.info("Paths initialized")
        
        # Get device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Initialize models with fp16
        logger.info("Initializing preprocess model...")
        preprocess_model = CropAndExtract(sadtalker_paths, device)
        if fp16_enabled:
            preprocess_model.net_recon = preprocess_model.net_recon.half()
        
        logger.info("Initializing audio to coeff model...")
        audio_to_coeff = Audio2Coeff(sadtalker_paths, device)
        if fp16_enabled:
            # Convert both sub-models to fp16
            audio_to_coeff.audio2pose_model = audio_to_coeff.audio2pose_model.half()
            audio_to_coeff.audio2exp_model = audio_to_coeff.audio2exp_model.half()
            # Convert the netG inside audio2exp_model
            audio_to_coeff.audio2exp_model.netG = audio_to_coeff.audio2exp_model.netG.half()
        
        logger.info("Initializing animation model...")
        animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)
        if fp16_enabled:
            # Convert all models in AnimateFromCoeff to fp16
            animate_from_coeff.generator = animate_from_coeff.generator.half()
            animate_from_coeff.kp_extractor = animate_from_coeff.kp_extractor.half()
            animate_from_coeff.he_estimator = animate_from_coeff.he_estimator.half()
            animate_from_coeff.mapping = animate_from_coeff.mapping.half()
        
        # Initialize SadTalker with preloaded models
        logger.info("Initializing SadTalker...")
        sad_talker = SadTalker(checkpoint_path='checkpoints', config_path='src/config', lazy_load=False)
        sad_talker.preprocess_model = preprocess_model
        sad_talker.audio_to_coeff = audio_to_coeff
        sad_talker.animate_from_coeff = animate_from_coeff
        
        # Set models to eval mode
        for model in [preprocess_model.net_recon, 
                     audio_to_coeff.audio2pose_model,
                     audio_to_coeff.audio2exp_model.netG,
                     animate_from_coeff.generator,
                     animate_from_coeff.kp_extractor,
                     animate_from_coeff.he_estimator,
                     animate_from_coeff.mapping]:
            model.eval()
            if fp16_enabled:
                model.half()
        
        logger.info("All models initialized successfully")
        
    except Exception as e:
        logger.error(f"Error during model initialization: {str(e)}", exc_info=True)
        raise

# Store active WebSocket connections
active_connections = set()

# Store generation tasks
generation_tasks = {}

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    active_connections.add(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except:
        active_connections.remove(websocket)

async def process_video(
    source_image: str, 
    audio_path: str, 
    websocket: Optional[WebSocket] = None
):
    """Process video generation and stream frames if websocket is provided"""
    try:
        # Generate unique ID for this generation
        gen_id = str(uuid.uuid4())
        generation_tasks[gen_id] = {"status": "processing"}
        
        logger.info(f"Starting video generation with image: {source_image} and audio: {audio_path}")
        
        # Create result directory
        result_dir = os.path.join("results", gen_id)
        os.makedirs(result_dir, exist_ok=True)
        
        # Process the video using preloaded models with fp16
        try:
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                result_path = sad_talker.test(
                    source_image=source_image,
                    driven_audio=audio_path,
                    preprocess='full',
                    still_mode=True,
                    use_enhancer=False,
                    batch_size=1,
                    size=256,
                    pose_style=0,
                    result_dir=result_dir
                )
            logger.info(f"Video generation completed. Result path: {result_path}")
        except Exception as e:
            logger.error(f"Error during video generation: {str(e)}", exc_info=True)
            raise
        
        # If websocket is provided, stream the video frames
        if websocket:
            try:
                cap = cv2.VideoCapture(result_path)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    # Convert frame to base64
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Send frame through websocket
                    await websocket.send_json({
                        "type": "frame",
                        "data": frame_base64
                    })
                    
                    # Small delay to control frame rate
                    await asyncio.sleep(0.03)  # ~30 fps
                    
                cap.release()
            except Exception as e:
                logger.error(f"Error during frame streaming: {str(e)}", exc_info=True)
                raise
            
        generation_tasks[gen_id] = {
            "status": "completed",
            "result_path": result_path
        }
        
        return result_path
        
    except Exception as e:
        logger.error(f"Error in process_video: {str(e)}", exc_info=True)
        generation_tasks[gen_id] = {
            "status": "error",
            "error": str(e)
        }
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate_video(
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
    stream: bool = False
):
    """Generate talking face video from image and audio"""
    temp_dir = None
    try:
        logger.info(f"Received request to generate video with image: {image.filename} and audio: {audio.filename}")
        
        # Create temporary directory for uploads
        temp_dir = os.path.join("temp", str(uuid.uuid4()))
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save uploaded files
        image_path = os.path.join(temp_dir, image.filename)
        audio_path = os.path.join(temp_dir, audio.filename)
        
        logger.info(f"Saving files to {image_path} and {audio_path}")
        
        # Read and preprocess image using cv2
        image_content = await image.read()
        nparr = np.frombuffer(image_content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Convert to RGB and save
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(image_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        
        # Save audio file
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)
            
        # Generate video
        result_path = await process_video(image_path, audio_path)
        
        logger.info(f"Video generation successful. Returning file: {result_path}")
        
        # Clean up temp files
        shutil.rmtree(temp_dir)
        
        # Return video file
        return FileResponse(
            result_path,
            media_type="video/mp4",
            filename="generated_video.mp4"
        )
        
    except Exception as e:
        logger.error(f"Error in generate_video endpoint: {str(e)}", exc_info=True)
        # Clean up on error
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/generate/stream")
async def generate_video_stream(websocket: WebSocket):
    """Generate and stream talking face video frames through WebSocket"""
    await websocket.accept()
    try:
        # Receive image and audio data from WebSocket
        image_data = await websocket.receive_bytes()
        audio_data = await websocket.receive_bytes()
        
        # Create temporary directory for uploads
        temp_dir = os.path.join("temp", str(uuid.uuid4()))
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save uploaded files
        image_path = os.path.join(temp_dir, "image.png")
        audio_path = os.path.join(temp_dir, "audio.wav")
        
        with open(image_path, "wb") as f:
            f.write(image_data)
        with open(audio_path, "wb") as f:
            f.write(audio_data)
            
        # Generate and stream video
        await process_video(image_path, audio_path, websocket)
        
        # Clean up temp files
        shutil.rmtree(temp_dir)
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        await websocket.send_json({
            "type": "error",
            "error": str(e)
        })
        await websocket.close()

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """Get status of a video generation task"""
    if task_id not in generation_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return generation_tasks[task_id]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
