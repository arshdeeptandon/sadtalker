from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException, WebSocketDisconnect, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import asyncio
from typing import Optional, Dict, List
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
from pydub import AudioSegment
import tempfile
from concurrent.futures import ThreadPoolExecutor
import queue
import time
import subprocess
import sys
from contextlib import contextmanager
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Add static file serving for examples directory
app.mount("/examples", StaticFiles(directory="examples"), name="examples")

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

# Add a context manager to temporarily suppress stdout
@contextmanager
def suppress_stdout():
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout

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
        sadtalker_paths = init_path('checkpoints', os.path.join(current_root_path, 'src/config'), VIDEO_SIZE, False, 'full')
        logger.info("Paths initialized")
        
        # Get device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Initialize models with fp16
        logger.info("Initializing preprocess model...")
        with suppress_stdout():  # Suppress model initialization prints
            preprocess_model = CropAndExtract(sadtalker_paths, device)
            if fp16_enabled:
                preprocess_model.net_recon = preprocess_model.net_recon.half()
            
            logger.info("Initializing audio to coeff model...")
            audio_to_coeff = Audio2Coeff(sadtalker_paths, device)
            if fp16_enabled:
                audio_to_coeff.audio2pose_model = audio_to_coeff.audio2pose_model.half()
                audio_to_coeff.audio2exp_model = audio_to_coeff.audio2exp_model.half()
                audio_to_coeff.audio2exp_model.netG = audio_to_coeff.audio2exp_model.netG.half()
            
            logger.info("Initializing animation model...")
            animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)
            if fp16_enabled:
                animate_from_coeff.generator = animate_from_coeff.generator.half()
                animate_from_coeff.kp_extractor = animate_from_coeff.kp_extractor.half()
                animate_from_coeff.he_estimator = animate_from_coeff.he_estimator.half()
                animate_from_coeff.mapping = animate_from_coeff.mapping.half()
            
            # Initialize SadTalker with preloaded models
            logger.info("Initializing SadTalker...")
            sad_talker = SadTalker(checkpoint_path='checkpoints', config_path='src/config', lazy_load=True)
            # Set the preloaded models in the sad_talker instance
            sad_talker.preprocess_model = preprocess_model
            sad_talker.audio_to_coeff = audio_to_coeff
            sad_talker.animate_from_coeff = animate_from_coeff
        
        # Set models to eval mode and move to device
        for model in [preprocess_model.net_recon, 
                     audio_to_coeff.audio2pose_model,
                     audio_to_coeff.audio2exp_model.netG,
                     animate_from_coeff.generator,
                     animate_from_coeff.kp_extractor,
                     animate_from_coeff.he_estimator,
                     animate_from_coeff.mapping]:
            model.eval()
            model.to(device)
            if fp16_enabled:
                model.half()
        
        logger.info("All models initialized successfully")
        
    except Exception as e:
        logger.error(f"Error during model initialization: {str(e)}", exc_info=True)
        raise

# Store active WebSocket connections and their state
active_connections = set()
generation_tasks = {}
CHUNK_DURATION = 3  # Duration of each audio chunk in seconds
VIDEO_SIZE = 256  # Video size in pixels
MAX_CONCURRENT_CHUNKS = 2  # Maximum number of chunks to process concurrently

# Thread pool for parallel processing
thread_pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_CHUNKS)

# Queue for managing chunk processing
chunk_queues: Dict[str, queue.Queue] = {}

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
                result_path = sad_talker.test_with_preloaded_models(
                    source_image=source_image,
                    driven_audio=audio_path,
                    preprocess='crop',
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

async def process_chunks_parallel(client_id: str, audio_path: str, image_path: str, websocket: WebSocket):
    """Process audio chunks in parallel"""
    try:
        # Load audio file
        audio = AudioSegment.from_file(audio_path)
        chunk_length_ms = CHUNK_DURATION * 1000
        total_chunks = len(audio) // chunk_length_ms + (1 if len(audio) % chunk_length_ms > 0 else 0)
        
        # Create queue for this client
        chunk_queues[client_id] = queue.Queue()
        
        # Create tasks for all chunks
        chunk_tasks = []
        for i in range(total_chunks):
            start_ms = i * chunk_length_ms
            end_ms = min((i + 1) * chunk_length_ms, len(audio))
            
            # Extract chunk
            chunk = audio[start_ms:end_ms]
            chunk_path = os.path.join("temp", client_id, f"audio_chunk_{i}.wav")
            chunk.export(chunk_path, format="wav")
            
            # Add chunk to queue
            chunk_queues[client_id].put((i, chunk_path, total_chunks))
        
        # Start processing chunks
        while not chunk_queues[client_id].empty():
            # Get next chunk to process
            chunk_index, chunk_path, total_chunks = chunk_queues[client_id].get()
            
            # Submit chunk for processing and wait for it to complete
            try:
                future = thread_pool.submit(
                    process_chunk_sync,
                    client_id,
                    image_path,
                    chunk_path,
                    chunk_index,
                    total_chunks
                )
                
                # Wait for chunk processing to complete with timeout
                result_path = future.result(timeout=60)  # Increased timeout to 60 seconds
                
                if not result_path or not os.path.exists(result_path):
                    raise FileNotFoundError(f"Generated video file not found for chunk {chunk_index}")
                
                # Send processing complete message for this chunk
                await websocket.send_json({
                    "type": "chunk_ready",
                    "chunk_index": chunk_index,
                    "total_chunks": total_chunks,
                    "file_path": result_path
                })
                
                # Now stream the frames
                await stream_chunk_frames(client_id, chunk_index, result_path, total_chunks, websocket)
                
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_index}: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "error": f"Error processing chunk {chunk_index}: {str(e)}",
                    "chunk_index": chunk_index
                })
                raise
        
        # Send completion message
        await websocket.send_json({
            "type": "processing_complete",
            "message": "All chunks processed successfully"
        })
        
    except Exception as e:
        logger.error(f"Error in parallel processing: {str(e)}", exc_info=True)
        raise
    finally:
        if client_id in chunk_queues:
            del chunk_queues[client_id]

def verify_video_audio(video_path: str) -> bool:
    """Verify if video has audio track"""
    try:
        # Use ffprobe to check audio stream
        cmd = [
            'ffprobe', 
            '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_type',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        has_audio = 'audio' in result.stdout.lower()
        logger.info(f"Video {video_path} audio check: {'Has audio' if has_audio else 'No audio'}")
        return has_audio
    except Exception as e:
        logger.error(f"Error checking video audio: {str(e)}")
        return False

def process_chunk_sync(client_id: str, source_image: str, audio_chunk_path: str, chunk_index: int, total_chunks: int):
    """Process a single chunk synchronously (called from thread pool)"""
    try:
        # Create result directory for this chunk
        result_dir = os.path.join("results", client_id, f"chunk_{chunk_index}")
        os.makedirs(result_dir, exist_ok=True)
        
        # Create a copy of the source image for this chunk
        chunk_image_path = os.path.join(result_dir, "source_image.png")
        shutil.copy2(source_image, chunk_image_path)
        
        # Process the chunk using preloaded models with fp16
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            result_path = sad_talker.test(
                source_image=chunk_image_path,
                driven_audio=audio_chunk_path,
                preprocess='full',
                still_mode=True,
                use_enhancer=False,
                batch_size=1,
                size=VIDEO_SIZE,
                pose_style=0,
                result_dir=result_dir
            )
            
        # Verify the file exists and is complete
        if not result_path or not os.path.exists(result_path):
            raise FileNotFoundError(f"Generated video file not found at {result_path}")
            
        # Wait for file to be fully written
        file_size = -1
        current_size = os.path.getsize(result_path)
        while file_size != current_size:
            file_size = current_size
            time.sleep(0.1)
            current_size = os.path.getsize(result_path)
            
        # Verify audio presence
        has_audio = verify_video_audio(result_path)
        if not has_audio:
            logger.warning(f"Generated video {result_path} has no audio track!")
            
        logger.info(f"Chunk {chunk_index}/{total_chunks} processed. Result path: {result_path}")
        return result_path
        
    except Exception as e:
        logger.error(f"Error processing chunk {chunk_index}: {str(e)}", exc_info=True)
        raise

@app.get("/audio/chunk/{client_id}/{chunk_index}")
async def get_chunk_audio(client_id: str, chunk_index: int):
    """Get audio stream for a specific chunk"""
    try:
        # Find the video file
        chunk_dir = os.path.join("results", client_id, f"chunk_{chunk_index}")
        video_files = []
        for root, _, files in os.walk(chunk_dir):
            for file in files:
                if file.endswith('.mp4'):
                    video_files.append(os.path.join(root, file))
        
        if not video_files:
            raise HTTPException(status_code=404, detail="No video file found")
            
        video_path = video_files[0]
        
        # Extract audio using ffmpeg
        temp_audio_path = os.path.join("temp", f"{client_id}_chunk_{chunk_index}_audio.mp3")
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'libmp3lame',  # Use MP3 codec
            '-ab', '192k',  # Audio bitrate
            temp_audio_path
        ]
        subprocess.run(cmd, capture_output=True)
        
        if not os.path.exists(temp_audio_path):
            raise HTTPException(status_code=500, detail="Failed to extract audio")
            
        return FileResponse(
            temp_audio_path,
            media_type="audio/mpeg",
            filename=f"chunk_{chunk_index}_audio.mp3",
            headers={
                "X-Chunk-Index": str(chunk_index),
                "X-Client-ID": client_id
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting chunk audio: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp audio file
        if os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except:
                pass

async def stream_chunk_frames(client_id: str, chunk_index: int, video_path: str, total_chunks: int, websocket: WebSocket):
    """Stream frames for a processed chunk"""
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at {video_path}")
            
        # Verify file is readable
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file {video_path}")
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Send audio URL before starting frame stream
        audio_url = f"/audio/chunk/{client_id}/{chunk_index}"
        await websocket.send_json({
            "type": "audio_ready",
            "audio_url": audio_url,
            "chunk_index": chunk_index,
            "fps": fps,
            "frame_count": frame_count
        })
            
        try:
            frame_index = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Resize frame to target size
                frame = cv2.resize(frame, (VIDEO_SIZE, VIDEO_SIZE))
                
                # Convert frame to base64 with optimized quality
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Calculate timestamp for audio sync
                timestamp = frame_index / fps
                
                # Send frame through websocket
                await websocket.send_json({
                    "type": "frame",
                    "data": frame_base64,
                    "chunk_index": chunk_index,
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                    "progress": {
                        "current": chunk_index + 1,
                        "total": total_chunks,
                        "percentage": int(((chunk_index + 1) / total_chunks) * 100)
                    }
                })
                
                frame_index += 1
                # Small delay to control frame rate
                await asyncio.sleep(1/fps)  # Use actual video FPS
                
        finally:
            cap.release()
            
    except Exception as e:
        logger.error(f"Error streaming chunk {chunk_index}: {str(e)}", exc_info=True)
        raise

@app.websocket("/ws/generate_video_stream")
async def websocket_generate_video(websocket: WebSocket):
    """WebSocket endpoint for video generation and streaming.
    
    Expected message sequence:
    1. First message: Image file as bytes
    2. Second message: Audio file as bytes
    
    The endpoint will then:
    1. Process the video generation
    2. Stream the generated frames
    3. Send audio URLs for synchronization
    """
    try:
        logger.info("WebSocket connection request received for /ws/generate_video_stream")
        await websocket.accept()
        client_id = str(uuid.uuid4())
        active_connections.add(websocket)
        temp_dir = None
        result_dir = None
        
        logger.info(f"[{client_id}] WebSocket connection accepted")
        
        try:
            # Create temporary and result directories
            temp_dir = os.path.join("temp", client_id)
            result_dir = os.path.join("results", client_id)
            os.makedirs(temp_dir, exist_ok=True)
            os.makedirs(result_dir, exist_ok=True)
            logger.info(f"[{client_id}] Created directories: {temp_dir} and {result_dir}")
            
            # Wait for image data with timeout
            logger.info(f"[{client_id}] Waiting for image data...")
            try:
                image_data = await asyncio.wait_for(websocket.receive_bytes(), timeout=30.0)
                logger.info(f"[{client_id}] Received image data: {len(image_data)} bytes")
                logger.info(f"[{client_id}] Image data first 100 bytes: {image_data[:100]}")
            except asyncio.TimeoutError:
                logger.error(f"[{client_id}] Timeout waiting for image data")
                raise
            except Exception as e:
                logger.error(f"[{client_id}] Error receiving image data: {str(e)}")
                raise
            
            # Save image
            image_path = os.path.join(temp_dir, "source_image.png")
            try:
                with open(image_path, "wb") as f:
                    f.write(image_data)
                logger.info(f"[{client_id}] Saved source image to: {image_path}")
                # Verify image was saved correctly
                if os.path.exists(image_path):
                    logger.info(f"[{client_id}] Verified image file exists, size: {os.path.getsize(image_path)} bytes")
                else:
                    raise FileNotFoundError(f"Image file was not saved correctly at {image_path}")
            except Exception as e:
                logger.error(f"[{client_id}] Error saving image file: {str(e)}")
                raise
            
            # Wait for audio data with timeout
            logger.info(f"[{client_id}] Waiting for audio data...")
            try:
                audio_data = await asyncio.wait_for(websocket.receive_bytes(), timeout=30.0)
                logger.info(f"[{client_id}] Received audio data: {len(audio_data)} bytes")
                logger.info(f"[{client_id}] Audio data first 100 bytes: {audio_data[:100]}")
            except asyncio.TimeoutError:
                logger.error(f"[{client_id}] Timeout waiting for audio data")
                raise
            except Exception as e:
                logger.error(f"[{client_id}] Error receiving audio data: {str(e)}")
                raise
            
            # Save audio
            audio_path = os.path.join(temp_dir, "audio.wav")
            try:
                with open(audio_path, "wb") as f:
                    f.write(audio_data)
                logger.info(f"[{client_id}] Saved audio file to: {audio_path}")
                # Verify audio was saved correctly
                if os.path.exists(audio_path):
                    logger.info(f"[{client_id}] Verified audio file exists, size: {os.path.getsize(audio_path)} bytes")
                else:
                    raise FileNotFoundError(f"Audio file was not saved correctly at {audio_path}")
            except Exception as e:
                logger.error(f"[{client_id}] Error saving audio file: {str(e)}")
                raise
            
            # Verify audio file
            try:
                audio = AudioSegment.from_file(audio_path)
                logger.info(f"[{client_id}] Audio loaded successfully: {len(audio)}ms duration, {audio.channels} channels, {audio.frame_rate}Hz")
            except Exception as e:
                logger.error(f"[{client_id}] Error loading audio file: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "error": f"Error loading audio file: {str(e)}"
                })
                return
            
            # Send processing started message
            await websocket.send_json({
                "type": "processing_started",
                "message": "Starting video generation"
            })
            logger.info(f"[{client_id}] Sent processing started message")
            
            # Process the video
            try:
                logger.info(f"[{client_id}] Starting video generation with image: {image_path} and audio: {audio_path}")
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    result_path = await process_video(image_path, audio_path, websocket)
                
                logger.info(f"[{client_id}] Video generation completed. Result path: {result_path}")
                
                # Send completion message
                await websocket.send_json({
                    "type": "complete",
                    "message": "Video generation completed",
                    "result_path": result_path
                })
                logger.info(f"[{client_id}] Sent completion message")
                
            except Exception as e:
                logger.error(f"[{client_id}] Error during video generation: {str(e)}", exc_info=True)
                await websocket.send_json({
                    "type": "error",
                    "error": str(e)
                })
                raise
            
        except asyncio.TimeoutError:
            logger.error(f"[{client_id}] Timeout waiting for files")
            await websocket.send_json({
                "type": "error",
                "error": "Timeout waiting for files"
            })
        except Exception as e:
            logger.error(f"[{client_id}] Error processing request: {str(e)}", exc_info=True)
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
            raise
            
    except WebSocketDisconnect:
        logger.info(f"[{client_id}] Client disconnected")
    except Exception as e:
        logger.error(f"[{client_id}] WebSocket error: {str(e)}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
        except:
            pass
    finally:
        active_connections.remove(websocket)
        # Clean up temp files
        try:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info(f"[{client_id}] Cleaned up temp directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"[{client_id}] Error cleaning up temp directory: {str(e)}")
        await websocket.close()
        logger.info(f"[{client_id}] WebSocket connection closed")

@app.get("/audio/{client_id}")
async def get_audio(client_id: str):
    """Endpoint to serve the generated audio"""
    try:
        audio_path = os.path.join("results", client_id, "audio.wav")
        if not os.path.exists(audio_path):
            # Try to find the audio in the video file
            video_dir = os.path.join("results", client_id)
            for root, _, files in os.walk(video_dir):
                for file in files:
                    if file.endswith('.mp4'):
                        video_path = os.path.join(root, file)
                        # Extract audio from video
                        temp_audio_path = os.path.join("temp", f"{client_id}_audio.wav")
                        try:
                            cmd = [
                                'ffmpeg', '-y',
                                '-i', video_path,
                                '-vn',  # No video
                                '-acodec', 'pcm_s16le',  # Use WAV codec
                                temp_audio_path
                            ]
                            subprocess.run(cmd, capture_output=True, check=True)
                            if os.path.exists(temp_audio_path):
                                # Move to results directory
                                shutil.move(temp_audio_path, audio_path)
                                break
                        except Exception as e:
                            logger.warning(f"Failed to extract audio from video: {str(e)}")
                            if os.path.exists(temp_audio_path):
                                os.remove(temp_audio_path)
            
            if not os.path.exists(audio_path):
                raise HTTPException(status_code=404, detail="Audio file not found")
        
        # Verify file is readable
        try:
            with open(audio_path, 'rb') as f:
                # Read first few bytes to verify file is readable
                f.read(1024)
        except Exception as e:
            logger.error(f"Error reading audio file: {str(e)}")
            raise HTTPException(status_code=500, detail="Error reading audio file")
            
        # Get file size
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            logger.error(f"Audio file is empty: {audio_path}")
            raise HTTPException(status_code=500, detail="Audio file is empty")
            
        logger.info(f"Serving audio file: {audio_path} (size: {file_size} bytes)")
        
        return FileResponse(
            audio_path, 
            media_type="audio/wav",
            headers={
                "Content-Disposition": f'attachment; filename="audio_{client_id}.wav"',
                "Content-Length": str(file_size),
                "Accept-Ranges": "bytes",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "Range",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving audio: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """Get status of a video generation task"""
    if task_id not in generation_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return generation_tasks[task_id]

# Add a general status endpoint
@app.get("/status")
async def get_general_status():
    """Get general server status"""
    return {
        "status": "running",
        "active_connections": len(active_connections),
        "active_tasks": len(generation_tasks),
        "server_time": time.strftime("%Y-%m-%d %H:%M:%S")
    }

@app.get("/download/chunk/{client_id}/{chunk_index}")
async def download_chunk(client_id: str, chunk_index: int):
    """Download a specific video chunk"""
    try:
        # Try the exact path we know exists
        exact_path = os.path.join("results", client_id, f"chunk_{chunk_index}", 
                                 "3e322d5b-b581-4396-874a-34240ef853f3",
                                 "source_image##audio_chunk_0_full.mp4")
        logger.info(f"Trying exact path: {exact_path}")
        
        if os.path.exists(exact_path):
            logger.info(f"Found file at exact path: {exact_path}")
            try:
                has_audio = verify_video_audio(exact_path)
                logger.info(f"Video has audio: {has_audio}")
            except Exception as e:
                logger.warning(f"Could not verify audio: {str(e)}")
                has_audio = False
                
            return FileResponse(
                exact_path,
                media_type="video/mp4",
                filename=f"chunk_{chunk_index}.mp4",
                headers={
                    "X-Has-Audio": str(has_audio).lower(),
                    "X-Chunk-Index": str(chunk_index),
                    "X-Client-ID": client_id,
                    "X-File-Path": exact_path
                }
            )
            
        # If exact path not found, try searching in the directory
        chunk_dir = os.path.join("results", client_id, f"chunk_{chunk_index}")
        logger.info(f"Exact path not found, searching in directory: {chunk_dir}")
        
        if not os.path.exists(chunk_dir):
            error_msg = f"Chunk directory not found: {chunk_dir}"
            logger.error(error_msg)
            raise HTTPException(status_code=404, detail=error_msg)
            
        # List all files and directories recursively
        def list_files_recursive(directory):
            files = []
            for root, dirs, filenames in os.walk(directory):
                for filename in filenames:
                    if filename.endswith('.mp4'):
                        full_path = os.path.join(root, filename)
                        files.append(full_path)
            return files
            
        all_video_files = list_files_recursive(chunk_dir)
        logger.info(f"Found video files recursively: {all_video_files}")
        
        if not all_video_files:
            error_msg = f"No video files found in {chunk_dir} or its subdirectories"
            logger.error(error_msg)
            raise HTTPException(status_code=404, detail=error_msg)
            
        # Use the first video file found
        video_path = all_video_files[0]
        logger.info(f"Using video file: {video_path}")
        
        try:
            has_audio = verify_video_audio(video_path)
            logger.info(f"Video has audio: {has_audio}")
        except Exception as e:
            logger.warning(f"Could not verify audio: {str(e)}")
            has_audio = False
            
        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename=f"chunk_{chunk_index}.mp4",
            headers={
                "X-Has-Audio": str(has_audio).lower(),
                "X-Chunk-Index": str(chunk_index),
                "X-Client-ID": client_id,
                "X-File-Path": video_path
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error downloading chunk: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/chunks/{client_id}")
async def list_chunks(client_id: str):
    """List all available chunks for a client"""
    try:
        base_dir = os.path.join("results", client_id)
        if not os.path.exists(base_dir):
            raise HTTPException(status_code=404, detail=f"Client directory not found: {base_dir}")
            
        chunks = []
        for chunk_dir in sorted(os.listdir(base_dir)):
            if chunk_dir.startswith("chunk_"):
                chunk_index = int(chunk_dir.split("_")[1])
                chunk_path = os.path.join(base_dir, chunk_dir)
                
                # List all files recursively
                def list_files_recursive(directory):
                    files = []
                    for root, dirs, filenames in os.walk(directory):
                        for filename in filenames:
                            if filename.endswith('.mp4'):
                                full_path = os.path.join(root, filename)
                                files.append(full_path)
                    return files
                
                video_files = list_files_recursive(chunk_path)
                logger.info(f"Found video files in {chunk_path}: {video_files}")
                
                if video_files:
                    video_path = video_files[0]  # Use first video file found
                    try:
                        has_audio = verify_video_audio(video_path)
                        size = os.path.getsize(video_path)
                    except Exception as e:
                        logger.warning(f"Error checking video {video_path}: {str(e)}")
                        has_audio = False
                        size = 0
                        
                    chunks.append({
                        "chunk_index": chunk_index,
                        "path": video_path,
                        "has_audio": has_audio,
                        "size": size,
                        "all_video_files": video_files
                    })
                    
        return JSONResponse(content={
            "client_id": client_id,
            "total_chunks": len(chunks),
            "chunks": chunks,
            "base_directory": base_dir
        })
        
    except Exception as e:
        error_msg = f"Error listing chunks: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
