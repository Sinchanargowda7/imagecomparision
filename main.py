import os
from PIL import Image
from src.engine import ImageComparator

def load_images_from_folder(folder_path, max_images=5):
    """Loads up to 'max_images' from a directory."""
    images = []
    names = []
    
    if not os.path.exists(folder_path):
        return [], []

    valid = ('.jpg', '.jpeg', '.png', '.webp')
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(valid)])
    
    for f in files[:max_images]:
        try:
            img = Image.open(os.path.join(folder_path, f))
            images.append(img)
            names.append(f)
        except:
            pass
            
    return images, names

def main():
    print("--- 🚀 Image Matcher CLI ---")
    
    # 1. Setup
    ref_path = "inputs/reference.jpg"
    cand_dir = "inputs/candidates"
    
    # 2. Validation
    if not os.path.exists(ref_path):
        print(f"❌ Error: Place your query image at '{ref_path}'")
        return

    # 3. Load
    print("⏳ Loading images...")
    ref_img = Image.open(ref_path)
    cand_imgs, cand_names = load_images_from_folder(cand_dir, max_images=10)
    
    if not cand_imgs:
        print(f"❌ Error: No images found in '{cand_dir}'")
        return

    print(f"✅ Loaded 1 Reference vs {len(cand_imgs)} Candidates.")

    # 4. Initialize Engine
    comp = ImageComparator()
    
    # 5. Run
    best_idx, best_score, all_scores = comp.find_best_match(ref_img, cand_imgs)
    
    # 6. Report
    print(f"\n🏆 WINNER: {cand_names[best_idx]}")
    print(f"📈 SCORE:  {best_score:.4f}")
    print("-" * 30)
    
    # Gap check in CLI
    sorted_s = sorted(all_scores, reverse=True)
    gap = sorted_s[0] - (sorted_s[1] if len(sorted_s) > 1 else 0)
    
    if gap < 0.05:
        print("⚠️ WARNING: Result is AMBIGUOUS (Low margin of victory).")
    else:
        print("✅ Result is Clear.")

if __name__ == "__main__":
    main()