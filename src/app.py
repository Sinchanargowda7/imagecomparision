import streamlit as st
from PIL import Image
from engine import ImageComparator

# --- 1. Helper: Human Score Conversion ---
def get_human_score(raw_score):
    """
    Converts Raw AI Score (Cosine Sim) to Human Percentage.
    REALITY CHECK: 
    - 0.80+ is usually an exact duplicate or extremely close match.
    - 0.60-0.80 is a strong semantic match (Same object, different angle).
    - < 0.40 is usually a non-match.
    """
    # We set the "Floor" to 0.40. This is calibrated for CLIP ViT-L-14.
    threshold = 0.40 

    if raw_score < threshold:
        # If score is very low, we show a low percentage
        return max(0, (raw_score / threshold) * 10)
    
    # Scale the valid range (0.40 to 1.0) to (0% to 100%)
    normalized = (raw_score - threshold) / (1 - threshold)
    return normalized * 100

# --- 2. Page Config ---
st.set_page_config(page_title="Visual Matcher Pro", page_icon="👁️", layout="wide")

# --- 3. Load Engine (Cached) ---
@st.cache_resource
def load_engine():
    return ImageComparator()

try:
    comparator = load_engine()
except Exception as e:
    st.error("Could not load the AI Model. Check logs.")
    st.stop()

# --- 4. UI Layout ---
st.title("👁️ Smart Image Comparator")
st.markdown("### Compare 1 Reference vs Multiple Candidates")
st.info("System Status: Calibrated for High-Accuracy Matching.")

col_ref, col_cand = st.columns([1, 2])

# --- ALLOW ALL IMAGE TYPES ---
all_image_types = ["jpg", "jpeg", "png", "webp", "jfif", "bmp", "tiff", "tif", "gif", "ico"]

# Inputs
with col_ref:
    st.header("1. Reference")
    ref_file = st.file_uploader("Upload Target Image", type=all_image_types, key="ref_uploader")
    if ref_file:
        ref_image = Image.open(ref_file)
        st.image(ref_image, caption="Query Object", width=300)

with col_cand:
    st.header("2. Candidates (Select 5+)")
    cand_files = st.file_uploader("Upload Database Images", type=all_image_types, accept_multiple_files=True, key="cand_uploader")
    
    candidate_images = []
    if cand_files:
        # Show mini grid
        cols = st.columns(5)
        for i, file in enumerate(cand_files):
            try:
                img = Image.open(file)
                candidate_images.append(img)
                with cols[i % 5]:
                    # --- FIXED: Used 'use_container_width' to remove yellow warnings ---
                    st.image(img, use_container_width=True) 
            except Exception as e:
                st.error(f"Error loading {file.name}")

# --- 5. Execution Logic ---
st.write("---")

if st.button("🚀 Run Comparison", type="primary"):
    if ref_file and cand_files:
        with st.spinner("🧠 Analyzing Features..."):
            
            # A. Run AI Analysis
            best_idx, best_raw, all_raw_scores = comparator.find_best_match(ref_image, candidate_images)
            
            # B. Gap Analysis (Checking for confusion)
            sorted_scores = sorted(all_raw_scores, reverse=True)
            winner_score = sorted_scores[0]
            runner_up = sorted_scores[1] if len(sorted_scores) > 1 else 0
            gap = winner_score - runner_up
            
            # C. Human Score Calculation
            human_score = get_human_score(best_raw)
            
            # --- DISPLAY RESULT ---
            st.subheader("🏆 Verdict")
            
            r_col1, r_col2 = st.columns([1, 2])
            
            with r_col1:
                st.image(candidate_images[best_idx], width=300, caption=f"Winner (Img #{best_idx+1})")
            
            with r_col2:
                # LOGIC: 
                # If we have multiple very high scores, it's not "Ambiguous" (bad), 
                # it's "Multiple Matches" (good).
                
                if human_score > 80:
                    st.success(f"✅ **EXACT MATCH FOUND**")
                    st.markdown(f"**Confidence:** {human_score:.1f}%")
                    
                    if gap < 0.05 and len(candidate_images) > 1:
                         st.info(f"ℹ️ Note: Several other images also matched closely.")
                    else:
                        st.write("The system is confident this is the specific object.")

                elif human_score > 60:
                    st.success(f"🔹 **STRONG MATCH**")
                    st.markdown(f"**Confidence:** {human_score:.1f}%")
                    st.write("This is likely the same object or category.")
                    
                    if gap < 0.05:
                        st.warning("⚠️ Multiple candidates look very similar to this one.")
                
                else:
                    st.error("❌ **NO CLEAR MATCH**")
                    st.write("The closest image is still quite different.")

                st.caption(f"Raw Score: {best_raw:.3f} | Margin over Runner-up: {gap:.3f}")

            # --- Details Table ---
            with st.expander("📊 See Analysis for All Candidates"):
                for i, score in enumerate(all_raw_scores):
                    h_s = get_human_score(score)
                    
                    # Visual Formatting
                    if i == best_idx:
                        label = f"🏆 WINNER (Img {i+1})"
                    else:
                        label = f"Candidate {i+1}"
                        
                    st.write(f"**{label}**")
                    st.progress(int(h_s), text=f"Match: {h_s:.1f}% (Raw: {score:.3f})")

    else:
        st.warning("⚠️ Please upload both a Reference image and Candidate images.")