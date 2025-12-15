import os
from PIL import Image
from src.engine import RealTimeSearchEngine

def main():
    print("--- 🚀 Starting Real-Time Engine ---")
    
    # 1. Initialize Engine
    engine = RealTimeSearchEngine()
    
    # 2. Check if DB needs seeding
    if len(engine.image_db) == 0:
        print("📭 Database is empty. Checking 'inputs/candidates'...")
        cand_dir = "inputs/candidates"
        
        if os.path.exists(cand_dir):
            # Scan EVERYTHING in the folder
            all_files = os.listdir(cand_dir)
            images = []
            
            print(f"📂 Scanning {len(all_files)} files in candidates folder...")
            
            for f in all_files:
                full_path = os.path.join(cand_dir, f)
                # Ignore folders, only check files
                if os.path.isfile(full_path):
                    try:
                        # Try to open as image
                        img = Image.open(full_path)
                        img.load() # Force load to verify
                        images.append(img)
                        print(f"  ✅ Loaded: {f}")
                    except:
                        # Skip text files, system files, etc.
                        pass
            
            if images:
                print(f"⚡ Indexing {len(images)} valid images...")
                engine.add_to_index(images)
                engine.save_db()
                print("✅ Database primed and saved!")
            else:
                print("⚠️ No valid image files found in candidates folder.")
        else:
            print("⚠️ 'inputs/candidates' folder not found.")
    else:
        print(f"✅ Database loaded with {len(engine.image_db)} existing items.")

    print("\n💡 Tip: Run 'streamlit run src/app.py' to use the Live Interface.")

if __name__ == "__main__":
    main()