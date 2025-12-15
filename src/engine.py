from sentence_transformers import SentenceTransformer, util
from PIL import Image
import torch

class ImageComparator:
    def __init__(self, model_name='clip-ViT-L-14'):
        """
        Initializes the AI Model.
        Using 'clip-ViT-L-14' for high accuracy.
        """
        print(f"⏳ Loading AI Model ({model_name})...")
        try:
            self.model = SentenceTransformer(model_name)
            print("✅ Model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e

    def find_best_match(self, query_img, candidate_images):
        """
        Compares 1 Query Image against a LIST of Candidate Images.
        Returns:
            best_idx (int): Index of the winning image.
            best_score (float): The highest raw score (0.0 to 1.0).
            all_scores (list): List of all scores for analysis.
        """
        try:
            # 1. Encode the Query Image (The one we are searching for)
            query_emb = self.model.encode(query_img, convert_to_tensor=True)

            # 2. Encode ALL Candidate Images (Batch Processing)
            candidate_embs = self.model.encode(candidate_images, convert_to_tensor=True)

            # 3. Calculate Cosine Similarity
            # Returns a list of scores, e.g., [0.1, 0.8, 0.4, 0.9]
            scores = util.cos_sim(query_emb, candidate_embs)[0]

            # 4. Extract Results
            best_score_idx = scores.argmax().item()
            best_score = scores[best_score_idx].item()

            return best_score_idx, best_score, scores.tolist()

        except Exception as e:
            print(f"❌ Error during comparison: {e}")
            return -1, 0.0, []