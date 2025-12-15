import streamlit as st
from PIL import Image
from engine import ImageComparator

# --- 1. Helper Function: Smart Human Scoring ---
def get_human_score(raw_score):
    """
    Converts 'Robot Math' (Cosine Similarity) into 'Human %'.
    
    The AI naturally sees all real-world photos as ~50% similar because 
    they share lighting, shapes, and physics. We need to 'floor' this.
    """
    # Anything below 0.60 raw score is effectively a "mismatch" for humans.
    threshold = 0.60 

    if raw_score < threshold:
        # If it's below 0.60, crush the score down to 0-10% range.
        # This fixes the issue of "Random images showing 50%"
        return max(0, (raw_score / threshold) * 10)
    
    # If it's above 0.60, scale it to spread across 0-100%
    # Example: Raw 0.80 -> Human 50% (Similar category)
    # Example: Raw 0.95 -> Human 95% (Exact Match)
    normalized = (raw_score - threshold) / (1 - threshold)
    return normalized * 100

# --- 2. Page Configuration ---
st.set_page_config(page_title="Smart Visual Matcher", page_icon="🎯", layout="wide")

# --- 3. Load the AI Engine (Cached) ---
@st.cache_resource
def load_engine():
    # This loads the heavy 'L-14' model you downloaded
    return ImageComparator()

comparator = load_engine()

# --- 4. Main UI Layout ---
st.title("🎯 Smart Image Recognition")
st.markdown("### Find the best match in a crowd.")
st.info("Logic Update: Scores are now adjusted so 'random' images show low % (0-10%) instead of 50%.")

st.markdown("---")

# Split screen into 2 columns
col_ref, col_cand = st.columns([1, 2])

# Allowed file types
allowed_types = ["jpg", "png", "jpeg", "webp", "jfif", "bmp", "tiff"]

# --- Left Column: Reference ---
with col_ref:
    st.header("1. Reference Image")
    st.write("The object you are looking for:")
    ref_file = st.file_uploader("Upload Reference", type=allowed_types, key="ref")
    
    if ref_file:
        ref_image = Image.open(ref_file)
        st.image(ref_image, caption="Query Object", width=300)

# --- Right Column: Candidates ---
with col_cand:
    st.header("2. Candidate Images")
    st.write("The database to search through:")
    cand_files = st.file_uploader("Upload Candidates (Select multiple)", type=allowed_types, accept_multiple_files=True, key="candidates")
    
    # Preview logic
    if cand_files:
        candidate_images = []
        # Create a mini grid to show uploads
        grid_cols = st.columns(5) 
        for idx, file in enumerate(cand_files):
            img = Image.open(file)
            candidate_images.append(img)
            # Show small thumbnail in grid
            with grid_cols[idx % 5]:
                st.image(img, width=100)

# --- 5. Execution Logic ---
st.markdown("---")

if st.button("🚀 Find Best Match", type="primary"):
    if ref_file and cand_files:
        with st.spinner("🧠 Analyzing vectors and textures..."):
            
            # 1. Run the AI (Get raw robot scores)
            best_idx, best_raw, all_raw_scores = comparator.find_best_match(ref_image, candidate_images)
            
            # 2. Convert to Human Score
            human_score = get_human_score(best_raw)
            
            # --- DISPLAY RESULTS ---
            st.subheader("🏆 The Verdict")
            
            # Layout for the winner
            win_col, text_col = st.columns([1, 2])
            
            with win_col:
                st.image(candidate_images[best_idx], width=300, caption=f"Winning Image (Index {best_idx+1})")
            
            with text_col:
                st.metric("Match Confidence", f"{human_score:.1f}%", delta=f"Raw AI Score: {best_raw:.3f}")
                
                # Dynamic feedback based on Human Score
                if human_score > 85:
                    st.success("✅ **EXACT MATCH:** This is the same object.")
                elif human_score > 60:
                    st.warning("⚠️ **SIMILAR:** Same category (e.g., both are dogs), but likely different objects.")
                else:
                    st.error("❌ **NO MATCH:** The closest image is still too different.")

            # --- Detailed Breakdown Table ---
            st.write("---")
            with st.expander("📊 See Analysis for All Candidates"):
                for i, raw_score in enumerate(all_raw_scores):
                    h_score = get_human_score(raw_score)
                    
                    # Highlight the winner
                    prefix = "🏆 WINNER" if i == best_idx else f"Candidate {i+1}"
                    
                    st.write(f"**{prefix}**")
                    st.progress(int(h_score), text=f"Human Score: {h_score:.1f}% (Raw: {raw_score:.3f})")

    else:
        st.warning("⚠️ Please upload a Reference Image AND at least one Candidate Image.")