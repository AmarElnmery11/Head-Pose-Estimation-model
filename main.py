import io
import tempfile
import os
import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse

from utils import predict_pose, process_video

# Initialize FastAPI app
app = FastAPI(
    title="Head Pose Estimation API",
    description="API to predict head pose (pitch, yaw, roll) from images and videos."
)

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    """
    Processes an uploaded image to estimate head pose and returns the image with axes drawn.
    """
    try:
        # Read the image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")
        
        # Use the function from utils.py
        output_image, error = predict_pose(image)
        if error:
            raise HTTPException(status_code=400, detail=error)

        # Encode the output image to a buffer
        _, buffer = cv2.imencode('.jpg', output_image)
        image_bytes = io.BytesIO(buffer)

        return StreamingResponse(image_bytes, media_type="image/jpeg")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    """
    Processes an uploaded video to estimate head pose in each frame,
    draws axes, and returns the processed video.
    """
    try:
        # Save the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
            contents = await file.read()
            temp_video_file.write(contents)
            temp_video_path = temp_video_file.name

        # Use the function from utils.py to process the video
        output_path, error = process_video(temp_video_path, "processed_video.mp4")
        
        # Clean up the temporary input file
        os.remove(temp_video_path)

        if error:
            raise HTTPException(status_code=400, detail=error)

        # Return the processed video file
        return FileResponse(output_path, media_type="video/mp4")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/home")
async def read_root():
    """
    Returns a simple greeting message.
    """
    return {"message": "Hello"}