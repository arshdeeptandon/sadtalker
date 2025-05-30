from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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
from src.generate_batch import get_data
from src.utils import audio
import scipy.io as scio
from src.facerender.modules.make_animation import keypoint_transformation

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

# Get the absolute path of the project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

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
        logger.info(f"Current root path: {current_root_path}")
        
        if not os.path.exists(os.path.join(current_root_path, 'src/config')):
            logger.error("src/config directory not found")
            raise RuntimeError("src/config directory not found")
            
        if not os.path.exists('checkpoints'):
            logger.error("checkpoints directory not found")
            raise RuntimeError("checkpoints directory not found")
            
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
                # First get the face coefficients
                logger.info("Extracting face coefficients...")
                first_frame_dir = os.path.join(result_dir, 'first_frame_dir')
                os.makedirs(first_frame_dir, exist_ok=True)
                first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
                    source_image, 
                    first_frame_dir, 
                    'full', 
                    True, 
                    512  # Increased from 256 to 512 for better quality
                )
                
                if first_coeff_path is None:
                    raise AttributeError("No face detected in the image")

                # Prepare audio data with mel spectrograms
                logger.info("Preparing audio data...")
                batch = get_data(
                    first_coeff_path=first_coeff_path,
                    audio_path=audio_path,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    ref_eyeblink_coeff_path=None,
                    still=False,
                    idlemode=False,
                    use_blink=True
                )

                # Get the coefficients first
                logger.info("Generating coefficients...")
                coeff_path = audio_to_coeff.generate(
                    batch=batch,
                    coeff_save_dir=result_dir,
                    pose_style=0
                )

                # Load the coefficients
                logger.info("Loading coefficients...")
                coeff_dict = scio.loadmat(coeff_path)
                coeffs = coeff_dict['coeff_3dmm']
                logger.info(f"Audio coefficients shape: {coeffs.shape}")

                # Load source semantics for additional coefficients
                source_semantics_dict = scio.loadmat(first_coeff_path)
                source_semantics_full = source_semantics_dict['coeff_3dmm']
                logger.info(f"Source semantics shape: {source_semantics_full.shape}")

                # Ensure source semantics has the right shape
                if len(source_semantics_full.shape) == 1:
                    source_semantics_full = source_semantics_full.reshape(1, -1)
                source_semantics_full = source_semantics_full[:1]  # Get first frame with all coefficients
                logger.info(f"Reshaped source semantics shape: {source_semantics_full.shape}")

                # Read and preprocess source image
                source_img = cv2.imread(source_image)
                if source_img is None:
                    raise ValueError(f"Could not read image: {source_image}")
                
                # Ensure proper color space conversion and normalization
                source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)  # Convert to RGB first
                source_img = cv2.resize(source_img, (512, 512), interpolation=cv2.INTER_LANCZOS4)  # Better quality resize
                source_img = source_img.astype(np.float32) / 255.0  # Normalize to [0, 1] in float32
                source_img = torch.from_numpy(source_img).permute(2, 0, 1).unsqueeze(0).to(batch['indiv_mels'].device)
                logger.info(f"Source image shape: {source_img.shape}")

                # Generate frames in batches
                predictions = []
                total_frames = coeffs.shape[0]
                batch_size = 20  # Process 20 frames at a time
                frames_buffer = []  # Buffer to store frames before streaming
                
                # Send initial progress
                if websocket:
                    await websocket.send_json({
                        "type": "status",
                        "message": "Generating frames...",
                        "progress": {
                            "current": 0,
                            "total": total_frames,
                            "percentage": 0
                        }
                    })

                for frame_idx in range(0, total_frames, batch_size):
                    end_idx = min(frame_idx + batch_size, total_frames)
                    current_batch_size = end_idx - frame_idx
                    
                    # Get current batch of coefficients
                    current_coeffs = coeffs[frame_idx:end_idx]
                    
                    # Generate predictions for current batch
                    with torch.no_grad():
                        batch_predictions = animate_from_coeff.generate(
                            source_img,
                            current_coeffs,
                            source_semantics_full,
                            batch['indiv_mels'][frame_idx:end_idx]
                        )
                    
                    # Store predictions
                    predictions.extend(batch_predictions)
                    
                    # Convert frames to numpy and store in buffer
                    for pred in batch_predictions:
                        frame = pred[0].permute(1, 2, 0).cpu().numpy()
                        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert back to BGR for video
                        frames_buffer.append(frame)
                    
                    # Calculate progress
                    progress_percentage = int((end_idx / total_frames) * 100)
                    
                    # Send progress update
                    if websocket:
                        await websocket.send_json({
                            "type": "status",
                            "message": "Generating frames...",
                            "progress": {
                                "current": end_idx,
                                "total": total_frames,
                                "percentage": progress_percentage
                            }
                        })
                        
                        # Start streaming after 40% of frames are generated
                        if progress_percentage >= 40 and frames_buffer:
                            # Stream buffered frames
                            for i, frame in enumerate(frames_buffer):
                                # Convert frame to JPEG with high quality
                                _, buffer = cv2.imencode('.jpg', frame, [
                                    cv2.IMWRITE_JPEG_QUALITY, 95,  # Increased quality
                                    cv2.IMWRITE_JPEG_OPTIMIZE, 1
                                ])
                                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                                
                                # Send frame with progress information
                                await websocket.send_json({
                                    "type": "frame",
                                    "data": frame_base64,
                                    "progress": {
                                        "current": frame_idx + i + 1,
                                        "total": total_frames,
                                        "percentage": int(((frame_idx + i + 1) / total_frames) * 100)
                                    }
                                })
                                
                                # Small delay to maintain smooth playback
                                await asyncio.sleep(0.03)  # ~30 fps
                            
                            # Clear buffer after streaming
                            frames_buffer = []

                # Stack all predictions
                predictions_ts = torch.stack(predictions, dim=1)
                logger.info(f"Final predictions shape: {predictions_ts.shape}")

                # Save the final video with high quality
                result_path = os.path.join(result_dir, "result.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(result_path, fourcc, 30.0, (512, 512))  # Increased resolution
                
                if not out.isOpened():
                    raise RuntimeError("Failed to create video writer")
                
                # Write all frames to video
                for i in range(predictions_ts.shape[1]):
                    frame = predictions_ts[0, i].permute(1, 2, 0).cpu().numpy()
                    frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame)
                out.release()

                # Verify the video file was created
                if not os.path.exists(result_path):
                    raise RuntimeError(f"Video file was not created at {result_path}")
                
                # Get file size to verify it's not empty
                file_size = os.path.getsize(result_path)
                if file_size == 0:
                    raise RuntimeError("Generated video file is empty")
                
                logger.info(f"Video saved successfully at {result_path} (size: {file_size} bytes)")

                # After video is complete, send completion message
                if websocket:
                    await websocket.send_json({
                        "type": "processing_complete",
                        "message": "Video generation completed",
                        "video_url": f"/results/{gen_id}/result.mp4"
                    })
                    
                logger.info(f"Video generation completed. Result path: {result_path}")
                
        except Exception as e:
            logger.error(f"Error during video generation: {str(e)}", exc_info=True)
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
    client_id = str(uuid.uuid4())
    active_connections.add(websocket)
    temp_dir = None  # Initialize temp_dir at the start
    
    try:
        # First receive the source image
        image_data = await websocket.receive_bytes()
        
        # Create temporary directory for this session
        temp_dir = os.path.join("temp", client_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save source image
        image_path = os.path.join(temp_dir, "source_image.png")
        with open(image_path, "wb") as f:
            f.write(image_data)
            
        # Send acknowledgment that we're ready for audio
        await websocket.send_json({
            "type": "ready_for_audio",
            "message": "Ready to receive audio"
        })
        
        # Receive full audio file
        audio_data = await websocket.receive_bytes()
        
        # Save audio file
        audio_path = os.path.join(temp_dir, "full_audio.wav")
        with open(audio_path, "wb") as f:
            f.write(audio_data)
            
        # Process video and stream frames using process_video
        await process_video(image_path, audio_path, websocket)
            
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error in generate_video_stream: {str(e)}", exc_info=True)
        if websocket.client_state.CONNECTED:
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
    finally:
        # Cleanup
        active_connections.remove(websocket)
        if temp_dir and os.path.exists(temp_dir):  # Check if temp_dir exists before trying to remove
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.error(f"Error cleaning up temp directory: {str(e)}")
        if websocket.client_state.CONNECTED:
            await websocket.close()

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """Get status of a video generation task"""
    if task_id not in generation_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return generation_tasks[task_id]

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler to log all errors"""
    logger.error(f"Global error handler caught: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__}
    )

@app.get("/")
async def get_index():
    """Serve the index page"""
    try:
        if not os.path.exists("static/index.html"):
            logger.error("static/index.html file not found")
            raise HTTPException(status_code=500, detail="index.html not found")
        return FileResponse("static/index.html")
    except Exception as e:
        logger.error(f"Error serving index.html: {str(e)}", exc_info=True)
        raise

@app.get("/results/{gen_id}/result.mp4", include_in_schema=True)
async def get_generated_video_direct(gen_id: str):
    """Serve the generated video file directly from results directory"""
    try:
        video_path = os.path.join(PROJECT_ROOT, "results", gen_id, "result.mp4")
        logger.info(f"Attempting to serve video from: {video_path}")
        
        if not os.path.exists(video_path):
            logger.error(f"Video not found at: {video_path}")
            raise HTTPException(status_code=404, detail="Video not found")
            
        file_size = os.path.getsize(video_path)
        if file_size == 0:
            logger.error(f"Video file is empty: {video_path}")
            raise HTTPException(status_code=404, detail="Video file is empty")
            
        logger.info(f"Serving video from: {video_path} (size: {file_size} bytes)")
        
        return FileResponse(
            path=video_path,
            media_type="video/mp4",
            filename="result.mp4",
            headers={
                "Accept-Ranges": "bytes",
                "Cache-Control": "no-cache",
                "Content-Type": "video/mp4"
            }
        )
    except Exception as e:
        logger.error(f"Error serving video: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results/{gen_id}/status", include_in_schema=True)
async def check_video_status_direct(gen_id: str):
    """Check video status directly from results directory"""
    logger.info(f"Status check requested for gen_id: {gen_id}")
    try:
        # Use absolute paths
        video_path = os.path.join(PROJECT_ROOT, "results", gen_id, "result.mp4")
        logger.info(f"Checking video path: {video_path}")
        
        if os.path.exists(video_path):
            file_size = os.path.getsize(video_path)
            logger.info(f"Found video at {video_path} with size {file_size} bytes")
            return JSONResponse({
                "exists": True,
                "path": video_path,
                "size": file_size,
                "url": f"/results/{gen_id}/result.mp4"
            })
        
        logger.warning(f"No video found for gen_id: {gen_id}")
        return JSONResponse({"exists": False})
    except Exception as e:
        logger.error(f"Error checking video status for {gen_id}: {str(e)}", exc_info=True)
        return JSONResponse({"exists": False, "error": str(e)})

@app.get("/static/results/{gen_id}/result.mp4")
async def get_generated_video(gen_id: str):
    """Serve the generated video file"""
    try:
        # Use absolute paths
        video_paths = [ 
            os.path.join(PROJECT_ROOT, "results", gen_id, "result.mp4"),
            os.path.join(PROJECT_ROOT, "static", "results", gen_id, "result.mp4")
        ]
        
        # Log all paths we're checking
        logger.info(f"Checking video paths for gen_id {gen_id}:")
        for path in video_paths:
            logger.info(f"Checking path: {path} (exists: {os.path.exists(path)})")
        
        for video_path in video_paths:
            if os.path.exists(video_path):
                # Verify file is not empty
                file_size = os.path.getsize(video_path)
                if file_size == 0:
                    logger.error(f"Video file exists but is empty: {video_path}")
                    continue
                    
                logger.info(f"Serving video from: {video_path} (size: {file_size} bytes)")
                
                # Read file into memory to ensure it's valid
                try:
                    with open(video_path, 'rb') as f:
                        content = f.read()
                        if len(content) == 0:
                            raise ValueError("File is empty")
                except Exception as e:
                    logger.error(f"Error reading video file: {str(e)}")
                    continue
                
                return Response(
                    content=content,
                    media_type="video/mp4",
                    headers={
                        "Content-Disposition": f"inline; filename=result.mp4",
                        "Content-Length": str(file_size),
                        "Accept-Ranges": "bytes",
                        "Cache-Control": "no-cache",
                        "Content-Type": "video/mp4"
                    }
                )
        
        logger.error(f"Video not found for gen_id: {gen_id}")
        raise HTTPException(status_code=404, detail="Video not found")
        
    except Exception as e:
        logger.error(f"Error serving video: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files after all routes are defined
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/examples", StaticFiles(directory="examples"), name="examples")
app.mount("/results", StaticFiles(directory="results", html=True), name="results")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
