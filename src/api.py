from fastapi import FastAPI, UploadFile, File
from engine import EnterpriseSearchEngine
from PIL import Image
import io
import uvicorn
import base64
import os

app = FastAPI(title="Visual Recognition API (Pinecone Edition)")

# Initialize Engine (Connects to Pinecone)
engine = EnterpriseSearchEngine()

def bytes_to_image(img_bytes) -> Image.Image:
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

@app.get("/health")
def health_check():
    # Ask Pinecone for status
    try:
        stats = engine.index.describe_index_stats()
        count = stats.total_vector_count
    except:
        count = "Unknown (Connecting...)"
    return {"status": "operational", "vectors_indexed": count}

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
    content = await file.read()
    query_image = bytes_to_image(content)
    
    # 1. Auto-Caption (BLIP)
    auto_caption = engine.generate_caption(query_image)
    
    # 2. Visual Search (Pinecone)
    _, score, metadata = engine.search_visual(query_image)
    
    match_data = None
    if metadata and 'path' in metadata:
        # Retrieve the image from local cache using the path stored in Cloud Metadata
        image_path = metadata['path']
        
        if os.path.exists(image_path):
            with open(image_path, "rb") as img_file:
                # Convert to Base64 for the UI
                encoded_string = base64.b64encode(img_file.read()).decode()
                match_data = {"base64": encoded_string, "score": score}

    return {
        "caption": auto_caption,
        "visual_match": match_data
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)