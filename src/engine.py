import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util
from PIL import Image

class RealTimeSearchEngine:
    def __init__(self, model_name='clip-ViT-L-14', db_path='vector_db.pkl'):
        """
        Initializes the AI with a persistent memory (Vector Database).
        """
        print(f"⏳ Loading Real-Time AI ({model_name})...")
        try:
            self.model = SentenceTransformer(model_name)
            self.db_path = db_path
            self.image_db = []      # Stores the actual images
            self.vector_db = None   # Stores the mathematical embeddings
            
            # Load existing database if it exists
            self.load_db()
            print("✅ Model & Database loaded.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e

    def add_to_index(self, images):
        """
        Encodes images ONCE and saves them to memory.
        This makes future searches instant.
        """
        if not images: return
        
        print(f"⚡ Indexing {len(images)} new items...")
        
        # 1. Encode all images at once (Batch processing)
        new_embeddings = self.model.encode(images, convert_to_tensor=True)
        new_embeddings = new_embeddings.cpu().numpy() # Move to CPU for storage

        # 2. Add to local memory
        if self.vector_db is None:
            self.vector_db = new_embeddings
            self.image_db = images
        else:
            self.vector_db = np.vstack((self.vector_db, new_embeddings))
            self.image_db.extend(images)
            
        print(f"✅ Knowledge Base now has {len(self.image_db)} items.")

    def search(self, query_img):
        """
        Real-time search: Encodes ONLY the query and compares against Memory.
        Returns: Best Match Index, Best Score, and All Scores.
        """
        if self.vector_db is None or len(self.image_db) == 0:
            return -1, 0.0, []

        # 1. Encode ONLY the query (Fast!)
        query_emb = self.model.encode(query_img, convert_to_tensor=True)
        
        # 2. Compare against the entire database instantly
        scores = util.cos_sim(query_emb, self.vector_db)[0]

        # 3. Find best match
        best_score_idx = scores.argmax().item()
        best_score = scores[best_score_idx].item()
        
        return best_score_idx, best_score, scores.tolist()

    def save_db(self):
        """Saves the memory to a file so we don't lose it on restart."""
        with open(self.db_path, 'wb') as f:
            pickle.dump({'vectors': self.vector_db, 'images': self.image_db}, f)
        print("💾 Database Saved to disk.")

    def load_db(self):
        """Loads memory from file."""
        if os.path.exists(self.db_path):
            with open(self.db_path, 'rb') as f:
                data = pickle.load(f)
                self.vector_db = data['vectors']
                self.image_db = data['images']
            print(f"📂 Restored {len(self.image_db)} items from disk.")