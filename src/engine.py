import os
import uuid
import time
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from typing import List, Tuple

class EnterpriseSearchEngine:
    def __init__(self):
        print(f"🚀 [System] Initializing Cloud Memory (Pinecone)...")
        
        # --- CONFIGURATION (PASTE KEY HERE) ---
        self.PINECONE_API_KEY = "pcsk_4q3XVV_UqhTyMduR97ZFyVoG15Tc6cKe4Xeiw3R8SeTczT6DMRxK7Q8nCFQruYvDewcdMs" 
        self.INDEX_NAME = "visual-search-prod"
        self.img_folder = "stored_images"
        
        # 1. Setup Local Image Cache (For display only)
        # We store the PIXELS locally, but the MATH goes to the cloud.
        if not os.path.exists(self.img_folder):
            os.makedirs(self.img_folder)

        # 2. Connect to Pinecone Cloud
        self.pc = Pinecone(api_key=self.PINECONE_API_KEY)
        
        # Check if index exists, if not create it
        existing_indexes = [i.name for i in self.pc.list_indexes()]
        
        if self.INDEX_NAME not in existing_indexes:
            print(f"☁️ Creating new Cloud Index: {self.INDEX_NAME}...")
            try:
                self.pc.create_index(
                    name=self.INDEX_NAME,
                    dimension=768, # CLIP ViT-L-14 outputs 768 dimensions
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                # Wait for cloud initialization
                time.sleep(10)
            except Exception as e:
                print(f"⚠️ Index creation warning: {e}")
            
        self.index = self.pc.Index(self.INDEX_NAME)

        # 3. Load AI Models
        print("   - Loading Visual Core (CLIP)...")
        self.search_model = SentenceTransformer('clip-ViT-L-14')
        
        print("   - Loading Caption Core (BLIP)...")
        self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # Get stats
        try:
            stats = self.index.describe_index_stats()
            count = stats.total_vector_count
        except:
            count = 0
            
        print(f"✅ [System] Cloud Online. Indexed Vectors: {count}")

    def ingest_images(self, images: List[Image.Image]) -> int:
        if not images: return 0
        
        # 1. Compute Embeddings
        embeddings = self.search_model.encode(images).tolist()
        
        vectors_to_upload = []
        
        # 2. Process Batch
        for i, img in enumerate(images):
            # Generate unique ID
            file_id = str(uuid.uuid4())
            filename = f"{file_id}.png"
            path = os.path.join(self.img_folder, filename)
            
            # Save pixels locally (cache)
            img.save(path, format="PNG")
            
            # Prepare Vector for Cloud (ID, Vector, Metadata)
            vectors_to_upload.append({
                "id": file_id,
                "values": embeddings[i],
                "metadata": {"path": path}
            })
            
        # 3. Upload to Cloud
        # Upsert: Update if exists, Insert if new
        self.index.upsert(vectors=vectors_to_upload)
        
        return len(vectors_to_upload)

    def search_visual(self, query_img: Image.Image) -> Tuple[int, float, dict]:
        # 1. Vector Search
        query_vec = self.search_model.encode(query_img).tolist()
        
        # Query Pinecone
        results = self.index.query(
            vector=query_vec,
            top_k=1,
            include_metadata=True
        )
        
        # Check results
        if not results['matches']:
            return -1, 0.0, None

        match = results['matches'][0]
        score = match['score'] # Pinecone returns Cosine Similarity (0-1)
        metadata = match['metadata']
        
        return 0, score, metadata

    def generate_caption(self, query_img: Image.Image) -> str:
        inputs = self.caption_processor(query_img, return_tensors="pt")
        out = self.caption_model.generate(**inputs, max_new_tokens=20)
        caption = self.caption_processor.decode(out[0], skip_special_tokens=True)
        return caption.capitalize()

    def persist_state(self):
        # Cloud auto-saves. No local pickle needed.
        pass