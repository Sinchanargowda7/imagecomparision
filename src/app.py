import streamlit as st
from PIL import Image
from engine import RealTimeSearchEngine

# --- 1. Helper: Human Score ---
def get_human_score(raw_score):
    threshold = 0.40
    if raw_score < threshold:
        return max(0, (raw_score / threshold) * 10)
    normalized = (raw_score - threshold) / (1 - threshold)
    return normalized * 100

# --- 2. Page Config ---
st.set_page_config(page_title="Real-Time Visual Search", page_icon="👁️", layout="wide")

# --- 3. Initialize Session State for Camera ---
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

# --- 4. Load Engine (Cached) ---
@st.cache_resource
def load_engine():
    return RealTimeSearchEngine()

try:
    engine = load_engine()
except Exception as e:
    st.error("Engine failed to start. Check logs.")
    st.stop()

# --- 5. SIDEBAR: Knowledge Base ---
with st.sidebar:
    st.header("🧠 Memory Bank")
    st.caption(f"Status: {len(engine.image_db)} images indexed")
    
    with st.expander("📂 Manage Database", expanded=False):
        st.write("### Add New Objects")
        new_files = st.file_uploader("Upload images to learn", accept_multiple_files=True)
        
        if new_files:
            if st.button("⚡ Memorize These Images", type="primary"):
                images = []
                for f in new_files:
                    try:
                        img = Image.open(f)
                        img.load()
                        images.append(img)
                    except:
                        st.warning(f"Skipped '{f.name}'")

                if images:
                    with st.spinner("Learning new patterns..."):
                        engine.add_to_index(images)
                    st.success(f"Added {len(images)} new items!")
                    st.rerun()
        
        st.divider()
        
        col_save, col_clear = st.columns(2)
        with col_save:
            if st.button("💾 Save"):
                engine.save_db()
                st.toast("Database saved!", icon="💾")
        with col_clear:
            if st.button("🗑️ Reset"):
                engine.image_db = []
                engine.vector_db = None
                st.warning("Memory wiped.")
                st.rerun()

# --- 6. MAIN PAGE: Search Interface ---
st.title("👁️ Real-Time Visual Search")

with st.expander("ℹ️ How to use"):
    st.write("""
    1. **Add Images:** Open the Sidebar menu to upload images you want the AI to remember.
    2. **Search:** Use the **Start Camera** button or **Upload** tab to find a match.
    3. **Results:** The AI will instantly find the most similar object from its memory.
    """)

tab_cam, tab_up = st.tabs(["📷 Live Camera", "📂 Upload Image"])

query_image = None

# --- TAB 1: Camera (With Start/Stop Buttons) ---
with tab_cam:
    st.write("### Camera Controls")
    
    # Create two columns for the buttons
    col_btn_start, col_btn_stop, col_spacer = st.columns([1, 1, 3])
    
    with col_btn_start:
        if st.button("Start Camera", type="primary", use_container_width=True):
            st.session_state.camera_active = True
            st.rerun() # Force reload to show camera immediately
            
    with col_btn_stop:
        if st.button("Stop Camera", use_container_width=True):
            st.session_state.camera_active = False
            st.rerun()

    st.write("---")

    # Only show the camera widget if the state is Active
    if st.session_state.camera_active:
        camera_file = st.camera_input("Take a photo")
        if camera_file:
            query_image = Image.open(camera_file)
    else:
        st.info("Camera is currently **OFF**. Press 'Start Camera' to begin.")

# --- TAB 2: File Upload ---
with tab_up:
    upload_file = st.file_uploader("Choose an image to search for...", label_visibility="visible")
    if upload_file:
        try:
            query_image = Image.open(upload_file)
        except:
            st.error("Invalid file type.")

# --- 7. Results Section ---
if query_image:
    st.divider()
    
    if len(engine.image_db) == 0:
        st.warning("⚠️ **Memory is Empty!** The AI doesn't know any images yet. Please open the Sidebar and add some images first.")
    else:
        # Search Logic
        best_idx, best_raw, all_scores = engine.search(query_image)
        
        # Scoring Logic
        human_score = get_human_score(best_raw)
        sorted_scores = sorted(all_scores, reverse=True)
        winner_score = sorted_scores[0]
        runner_up = sorted_scores[1] if len(sorted_scores) > 1 else 0
        gap = winner_score - runner_up
        match_img = engine.image_db[best_idx]

        # Display Logic
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.write("#### 🔎 Your Query")
            st.image(query_image, width=300, caption="What you looking for")
            
        with c2:
            st.write("#### 🎯 AI Result")
            
            container = st.container(border=True)
            with container:
                r_col_img, r_col_txt = st.columns([1, 2])
                
                with r_col_img:
                    st.image(match_img, use_container_width=True)
                
                with r_col_txt:
                    if human_score > 80:
                        st.success(f"**EXACT MATCH** ({human_score:.1f}%)")
                        st.caption("The AI is highly confident.")
                    elif human_score > 60:
                        st.info(f"**LIKELY MATCH** ({human_score:.1f}%)")
                        st.caption("Looks very similar.")
                    else:
                        st.error(f"**NO MATCH** ({human_score:.1f}%)")
                        st.caption("Nothing in the database looks like this.")
                    
                    if gap < 0.05 and len(engine.image_db) > 1:
                        st.warning("⚠️ Ambiguous: Multiple similar items found.")
        
        with st.expander("📊 View All Candidates"):
            for i, score in enumerate(all_scores):
                h_s = get_human_score(score)
                st.write(f"**Item #{i+1}**")
                st.progress(int(h_s), text=f"{h_s:.1f}% Similarity")