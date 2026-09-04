from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import os
import shutil
from ultralytics import YOLO

app = FastAPI(title="CCTV Detection System")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_videos")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Robust YOLO Model Loading
def get_working_model():
    model_path = os.path.join(BASE_DIR, "best.pt")
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        try:
            return YOLO(model_path)
        except Exception:
            print("Corrupt best.pt file detected. Falling back to yolov8n.pt...")
    
    return YOLO("yolov8n.pt")

model = get_working_model()

# Global Video Source variable (Default: Webcam = 0)
current_video_source = 0

def generate_frames():
    global current_video_source
    
    source = current_video_source
    if isinstance(source, str):
        source = source.strip('"').strip("'").replace('\\', '/')
        
    camera = cv2.VideoCapture(source)

    while True:
        success, frame = camera.read()
        if not success:
            if isinstance(source, str):
                # Video auto loop reset
                camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break
        
        # YOLO Detection Inference
        results = model.predict(frame, conf=0.45, verbose=False)
        annotated_frame = results[0].plot()

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(BASE_DIR, "templates", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>templates/index.html file not found!</h1>", status_code=404)

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    global current_video_source
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    current_video_source = file_path
    return HTMLResponse(content="Video uploaded and stream updated successfully", status_code=200)

@app.post("/set_webcam")
async def set_webcam():
    global current_video_source
    current_video_source = 0
    return HTMLResponse(content="Switched to Webcam", status_code=200)

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        generate_frames(), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)