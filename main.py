from src.engine import ImageComparator

def main():
    # 1. Initialize the engine (Loads the AI Model)
    # We do this outside the loop so it stays in memory (fast)
    comparator = ImageComparator()

    print("\n--- 🔍 AI Image Matcher initialized ---")
    print("Place your images in the 'inputs' folder.")

    # 2. Define your images here
    # Make sure these files exist in your 'inputs' folder!
    image_1 = "inputs/image_a.jpg" 
    image_2 = "inputs/image_b.jpg"

    print(f"\nComparing:\n1. {image_1}\n2. {image_2}")

    # 3. Run Comparison
    score = comparator.compare(image_1, image_2)
    
    # 4. Interpret Results
    print("-" * 30)
    print(f"📈 Similarity Score: {score:.4f} ({score*100:.1f}%)")
    print("-" * 30)

    if score > 0.85:
        print("✅ MATCH: These images are very likely the same object.")
    elif score > 0.65:
        print("⚠️ SIMILAR: These look similar, but might be variants.")
    else:
        print("❌ DIFFERENT: These images are not related.")

if __name__ == "__main__":
    main()