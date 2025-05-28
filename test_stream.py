import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
from PIL import Image
import io
import os
import uuid
import aiohttp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def upload_files(image_path, audio_path):
    """Upload image and audio files to the server"""
    url = "http://localhost:8000/generate/stream"
    
    # Read files
    with open(image_path, 'rb') as f:
        image_data = f.read()
    with open(audio_path, 'rb') as f:
        audio_data = f.read()
    
    # Create WebSocket connection
    async with websockets.connect(url) as websocket:
        # Send image and audio data
        await websocket.send(image_data)
        await websocket.send(audio_data)
        
        # Create output directory for frames
        output_dir = "test_stream_output"
        os.makedirs(output_dir, exist_ok=True)
        
        frame_count = 0
        try:
            while True:
                # Receive frame data
                response = await websocket.recv()
                data = json.loads(response)
                
                if data["type"] == "error":
                    logger.error(f"Error from server: {data['error']}")
                    break
                    
                if data["type"] == "frame":
                    # Decode base64 frame
                    frame_data = base64.b64decode(data["data"])
                    frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # Save frame
                        frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
                        cv2.imwrite(frame_path, frame)
                        logger.info(f"Saved frame {frame_count}")
                        frame_count += 1
                        
                        # Display frame (optional)
                        cv2.imshow('Streaming Test', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                            
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
        finally:
            cv2.destroyAllWindows()
            logger.info(f"Total frames received: {frame_count}")

async def main():
    """Main function to test streaming"""
    # Test files - using example files from SadTalker
    image_path = "examples/source_image/full3.png"  # Example image from SadTalker
    audio_path = "examples/driven_audio/RD_Radio31_000.wav"  # Example audio from SadTalker
    
    # Verify files exist
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        logger.info("Please make sure you are running the script from the SadTalker root directory")
        logger.info("Available example images:")
        example_dir = "examples/source_image"
        if os.path.exists(example_dir):
            for file in os.listdir(example_dir):
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    logger.info(f"- {os.path.join(example_dir, file)}")
        return
        
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        logger.info("Please make sure you are running the script from the SadTalker root directory")
        logger.info("Available example audio files:")
        audio_dir = "examples/driven_audio"
        if os.path.exists(audio_dir):
            for file in os.listdir(audio_dir):
                if file.endswith(('.wav', '.mp3')):
                    logger.info(f"- {os.path.join(audio_dir, file)}")
        return
        
    logger.info(f"Using image: {image_path}")
    logger.info(f"Using audio: {audio_path}")
    
    try:
        await upload_files(image_path, audio_path)
    except Exception as e:
        logger.error(f"Error during streaming test: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 
