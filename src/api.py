from fastapi import FastAPI, UploadFile, File
from engine import EnterpriseSearchEngine
from PIL import Image
import io
import uvicorn
import base64
import os
import uuid # Needed for temp file

app = FastAPI(title="Visual Recognition API (Hybrid)")

engine = EnterpriseSearchEngine()

def bytes_to_image(img_bytes) -> Image.Image:
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

def get_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None

@app.get("/health")
def health_check():
    try:
        stats = engine.index.describe_index_stats()
        count = stats.total_vector_count
        faces = len(engine.known_face_encodings)
    except:
        count = "Unknown"
        faces = 0
    return {"status": "operational", "vectors": count, "faces_loaded": faces}

@app.post("/ingest")
async def ingest_endpoint(files: list[UploadFile] = File(...)):
    images = []
    for file in files:
        content = await file.read()
        images.append(bytes_to_image(content))
    count = engine.ingest_images(images)
    return {"message": "Ingestion successful", "batch_size": count}

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    # 1. Save temp file (Needed for Face Recognition library)
    temp_filename = f"temp_{uuid.uuid4()}.jpg"
    content = await file.read()
    with open(temp_filename, "wb") as buffer:
        buffer.write(content)
        
    query_image = bytes_to_image(content)
    
    # 2. Run Brains
    caption = engine.generate_caption(query_image)
    
    # A. Face Search
    face_result = engine.search_face(temp_filename)
    if face_result:
        face_result['base64'] = get_base64(face_result['metadata']['path'])
        
    # B. Visual Search (CLIP) - Always run as fallback
    _, score, metadata = engine.search_visual(query_image)
    visual_result = None
    if metadata:
        visual_result = {"score": score, "base64": get_base64(metadata['path'])}

    # Cleanup
    if os.path.exists(temp_filename):
        os.remove(temp_filename)

    return {
        "caption": caption,
        "face_match": face_result,
        "visual_match": visual_result
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)