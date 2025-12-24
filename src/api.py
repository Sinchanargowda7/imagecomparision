from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import uuid
import base64
import face_recognition

from engine import EnterpriseSearchEngine

app = FastAPI()
engine = EnterpriseSearchEngine()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- HELPERS ---------------- #

def get_base64(path: str):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None


def is_image(file: UploadFile) -> bool:
    try:
        Image.open(io.BytesIO(file.file.read()))
        file.file.seek(0)
        return True
    except:
        file.file.seek(0)
        return False


# ---------------- ROUTES ---------------- #

@app.get("/health")
def health():
    return {
        "vectors": engine.index.describe_index_stats()["total_vector_count"],
        "faces_loaded": len(engine.known_face_encodings),
    }


@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    images = []

    for file in files:
        if not is_image(file):
            continue  # safely ignore non-images

        try:
            img = Image.open(io.BytesIO(await file.read())).convert("RGB")
            images.append(img)
        except:
            continue

    count = engine.ingest_images(images)
    return {"ingested": count}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save temp file
    temp_path = f"temp_{uuid.uuid4()}.jpg"
    img_bytes = await file.read()

    try:
        query_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        query_image.save(temp_path)
    except:
        return {
            "face_matches": [],
            "visual_match": None
        }

    # ---------------- FACE SEARCH ---------------- #
    face_matches = engine.search_faces(temp_path)

    # Detect if query has any face at all
    query_has_face = False
    try:
        img_np = face_recognition.load_image_file(temp_path)
        query_has_face = len(face_recognition.face_encodings(img_np)) > 0
    except:
        pass

    visual_match = None

    # FACE-GATED LOGIC
    if query_has_face:
        if face_matches:
            for f in face_matches:
                f["base64"] = get_base64(f["metadata"]["path"])
        # ❌ DO NOT RUN CLIP
    else:
        # No face → allow CLIP (animals, objects, scenes)
        _, score, metadata = engine.search_visual(query_image)
        if metadata:
            visual_match = {
                "score": score,
                "base64": get_base64(metadata["path"])
            }

    os.remove(temp_path)

    return {
        "face_matches": face_matches,
        "visual_match": visual_match
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
