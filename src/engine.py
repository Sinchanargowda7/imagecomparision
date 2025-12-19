import os
import uuid
import time
import shutil
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image, ImageOps
from typing import List, Tuple
import face_recognition

class EnterpriseSearchEngine:
    def __init__(self):
        print(f"🚀 [System] Initializing Engine...")
        
        # --- CONFIGURATION ---
        self.PINECONE_API_KEY = "pcsk_4q3XVV_UqhTyMduR97ZFyVoG15Tc6cKe4Xeiw3R8SeTczT6DMRxK7Q8nCFQruYvDewcdMs" 
        self.INDEX_NAME = "visual-search-prod"
        self.img_folder = "stored_images"
        
        os.makedirs(self.img_folder, exist_ok=True)

        # 1. Connect to Pinecone
        self.pc = Pinecone(api_key=self.PINECONE_API_KEY)
        existing_indexes = [i.name for i in self.pc.list_indexes()]
        
        if self.INDEX_NAME not in existing_indexes:
            try:
                self.pc.create_index(
                    name=self.INDEX_NAME,
                    dimension=768, 
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                time.sleep(5)
            except: pass
        self.index = self.pc.Index(self.INDEX_NAME)

        # 2. Load Models
        print("   - Loading CLIP...")
        self.search_model = SentenceTransformer('clip-ViT-L-14')
        
        print("   - Loading BLIP...")
        self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # 3. Load Faces (Safe Mode)
        print("   - Loading Face Memory...")
        self.known_face_encodings = []
        self.known_face_paths = []
        self.reload_faces()
        
        print(f"✅ System Ready. Faces: {len(self.known_face_encodings)}")

    def reload_faces(self):
        """Learns faces safely."""
        self.known_face_encodings = []
        self.known_face_paths = []
        
        if not os.path.exists(self.img_folder): return
        files = os.listdir(self.img_folder)
        
        for f in files:
            path = os.path.join(self.img_folder, f)
            try:
                # Safe Loader
                image = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(image)
                
                if len(encodings) > 0:
                    self.known_face_encodings.append(encodings[0])
                    self.known_face_paths.append(path)
            except Exception:
                pass 

    def ingest_images(self, images: List[Image.Image]) -> int:
        if not images: return 0
        
        print(f"📥 Ingesting {len(images)} images...")
        embeddings = self.search_model.encode(images).tolist()
        vectors = []
        
        for i, img in enumerate(images):
            file_id = str(uuid.uuid4())
            filename = f"{file_id}.jpg"
            path = os.path.join(self.img_folder, filename)
            
            try:
                img = img.convert("RGB")
                img.save(path, format="JPEG", quality=95)
                
                vectors.append({
                    "id": file_id,
                    "values": embeddings[i],
                    "metadata": {"path": path}
                })
            except Exception as e:
                print(f"   ❌ Save failed: {e}")
            
        if vectors:
            self.index.upsert(vectors=vectors)
            self.reload_faces()
        
        return len(vectors)

    def search_face(self, query_img_path: str) -> dict:
        if not self.known_face_encodings: return None
        try:
            image = face_recognition.load_image_file(query_img_path)
            unknown_encodings = face_recognition.face_encodings(image)
            
            if not unknown_encodings: return None
            
            query_face = unknown_encodings[0]
            distances = face_recognition.face_distance(self.known_face_encodings, query_face)
            best_match = None
            best_score = -1
            
            for i, distance in enumerate(distances):
                if distance < 0.55: 
                    score = 1 - distance
                    if score > best_score:
                        best_score = score
                        best_match = {"score": score, "metadata": {"path": self.known_face_paths[i]}}
            return best_match
        except:
            return None

    def search_visual(self, query_img: Image.Image) -> Tuple[int, float, dict]:
        """
        SMART SEARCH: Fetches top 10 matches and returns the first one that ACTUALLY EXISTS.
        """
        query_vec = self.search_model.encode(query_img).tolist()
        
        # 1. Fetch TOP 10 matches (The Backup Plan)
        results = self.index.query(vector=query_vec, top_k=10, include_metadata=True)
        
        if not results['matches']: return -1, 0.0, None

        # 2. Iterate through matches until we find a valid file
        for match in results['matches']:
            path = match['metadata']['path']
            
            # THE MAGIC FIX: Check if file exists before returning it
            if os.path.exists(path):
                print(f"✅ Found valid match: {path}")
                return 0, match['score'], match['metadata']
            else:
                print(f"⚠️ Match missing on disk (Skipping): {path}")

        # If loop finishes and nothing was found
        print("❌ All top 10 matches are missing from disk.")
        return -1, 0.0, None

    def generate_caption(self, query_img: Image.Image) -> str:
        inputs = self.caption_processor(query_img, return_tensors="pt")
        out = self.caption_model.generate(**inputs, max_new_tokens=20)
        return self.caption_processor.decode(out[0], skip_special_tokens=True).capitalize()