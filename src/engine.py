from sentence_transformers import SentenceTransformer, util
from PIL import Image

class ImageComparator:
    def __init__(self):
        # CHANGED: Switched to 'clip-ViT-L-14'.
        # It is slower but MUCH smarter than the previous B-32 model.
        print("⏳ Loading High-Accuracy Model (clip-ViT-L-14)...")
        self.model = SentenceTransformer('clip-ViT-L-14')
        print("✅ Model loaded.")

    def find_best_match(self, query_img, candidate_images):
        """
        Compares 1 Query Image against a LIST of Candidate Images.
        Returns the index and score of the best match.
        """
        try:
            # 1. Encode the Query Image (The one we are searching for)
            query_emb = self.model.encode(query_img, convert_to_tensor=True)

            # 2. Encode ALL Candidate Images at once
            candidate_embs = self.model.encode(candidate_images, convert_to_tensor=True)

            # 3. Compare Query vs All Candidates
            # This returns a list of scores, e.g., [0.1, 0.8, 0.4, 0.9]
            scores = util.cos_sim(query_emb, candidate_embs)[0]

            # 4. Find the highest score
            best_score_idx = scores.argmax().item()
            best_score = scores[best_score_idx].item()

            # Return all scores so we can show them to the user
            return best_score_idx, best_score, scores.tolist()

        except Exception as e:
            print(f"Error: {e}")
            return -1, 0.0, []