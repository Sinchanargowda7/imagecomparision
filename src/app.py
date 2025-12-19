import streamlit as st
import requests
from PIL import Image
import io
import base64
from serpapi import GoogleSearch

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8000"
SERPAPI_KEY = "0d3e917f0ddb66d5d989c8be3ae84302f3f819596119589eb279331be6f55e8e"

st.set_page_config(page_title="Enterprise Visual Search", page_icon="🏢", layout="wide")

# --- HELPERS ---
def base64_to_image(b64_str):
    if not b64_str:
        return None
    try:
        img_data = base64.b64decode(b64_str)
        return Image.open(io.BytesIO(img_data))
    except Exception:
        return None

# --- SESSION STATE ---
if "cam_on" not in st.session_state:
    st.session_state.cam_on = False
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "preview_buffer" not in st.session_state:
    st.session_state.preview_buffer = []

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ System Controls")
    try:
        status = requests.get(f"{API_URL}/health", timeout=5).json()
        vec_count = status.get("vectors", 0)
        face_count = status.get("faces_loaded", 0)
        st.success(f"🟢 API Online | Indexed: {vec_count}")
        st.caption(f"👤 Faces Loaded: {face_count}")
    except Exception:
        st.error("🔴 API Offline. Run backend first.")
        st.stop()

    st.divider()
    st.subheader("📥 Data Ingestion")
    ingest_mode = st.radio(
        "Source:", ["📂 Manual Upload", "🌐 Web Crawler (Google)"], horizontal=True
    )

    # --- MANUAL UPLOAD ---
    if ingest_mode == "📂 Manual Upload":
        files = st.file_uploader(
            "Drag & Drop Files",
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
        )
        if files and st.button("Transmit to Server", type="primary"):
            payload = [("files", (f.name, f, f.type)) for f in files]
            with st.spinner("Transmitting..."):
                res = requests.post(f"{API_URL}/ingest", files=payload)
                if res.status_code == 200:
                    st.toast("Ingestion Complete!", icon="✅")
                    st.session_state.uploader_key += 1
                    st.rerun()
                else:
                    st.error("Server failed.")

    # --- GOOGLE SEARCH INGESTION ---
    elif ingest_mode == "🌐 Web Crawler (Google)":
        with st.form("web_search_form"):
            topic = st.text_input("Topic", placeholder="e.g. Cat sitting on sofa")
            count = st.number_input("Count", 1, 50, 5)
            submitted = st.form_submit_button("🔍 Search Google", type="primary")

        if submitted and topic:
            st.session_state.preview_buffer = []
            params = {"q": topic, "tbm": "isch", "api_key": SERPAPI_KEY}
            search = GoogleSearch(params)
            results = search.get_dict()
            image_urls = [
                img["original"]
                for img in results.get("images_results", [])[:count]
            ]

            for i, url in enumerate(image_urls):
                try:
                    resp = requests.get(url, timeout=3)
                    if resp.status_code == 200:
                        img = Image.open(io.BytesIO(resp.content))
                        st.session_state.preview_buffer.append(
                            {
                                "id": i,
                                "bytes": resp.content,
                                "img": img,
                                "selected": True   # 👈 IMPORTANT
                            }
                        )

                except Exception:
                    pass

        if st.session_state.preview_buffer:
            with st.form("review_upload"):
                for idx, item in enumerate(st.session_state.preview_buffer):
                    st.image(item["img"], use_container_width=True)

                    keep = st.checkbox(
                        "Keep",
                        value=item["selected"],
                        key=f"keep_{item['id']}"
                    )
                    st.session_state.preview_buffer[idx]["selected"] = keep

                if st.form_submit_button("🚀 Upload"):
                    for item in st.session_state.preview_buffer:
                        if not item["selected"]:
                            continue  # 👈 SKIP UNCHECKED IMAGES

                        files_payload = {
                            "files": ("google.jpg", item["bytes"], "image/jpeg")
                        }
                        requests.post(f"{API_URL}/ingest", files=files_payload)

                    st.session_state.preview_buffer = []
                    st.rerun()


# --- MAIN UI ---
st.title("Visual Recognition")
tab1, tab2 = st.tabs(["📷 Live Feed", "📂 File Analysis"])
query_img = None

with tab1:
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("Start camera"):
        st.session_state.cam_on = True
    if c2.button("Stop camera"):
        st.session_state.cam_on = False
    if st.session_state.cam_on:
        cam = st.camera_input("Feed", label_visibility="collapsed")
        if cam:
            query_img = Image.open(cam)

with tab2:
    up = st.file_uploader("Upload Query Image")
    if up:
        query_img = Image.open(up)

# --- ANALYSIS ---
if query_img:
    buf = io.BytesIO()
    query_img.save(buf, format="PNG")
    api_files = {"file": ("query.png", buf.getvalue(), "image/png")}

    with st.spinner("Analyzing Scene..."):
        response = requests.post(f"{API_URL}/predict", files=api_files).json()

    caption = response.get("caption", "No description")
    face_match = response.get("face_match")
    visual_match = response.get("visual_match")

    c_in, c_out = st.columns([1, 2])

    with c_in:
        st.subheader("Input Source")
        st.image(query_img, width=300)

    with c_out:
        st.subheader("Intelligence Report")
        st.info(f"**AI Description:** {caption}")
        st.divider()

        # --- FACE MATCH ---
        if face_match:
            img = base64_to_image(face_match["base64"])
            if img:
                st.success("✅ Identity Confirmed")
                st.image(img, width=150)
                st.caption(f"Face similarity: {face_match['score']:.2f}")

        # --- VISUAL MATCH ---
        elif visual_match:
            raw_score = visual_match["score"]
            threshold = 0.15

            vc1, vc2 = st.columns([1, 2])
            with vc1:
                db_img = base64_to_image(visual_match["base64"])
                if db_img:
                    st.image(db_img, width=150)
                else:
                    st.error("Image missing")

            with vc2:
                if raw_score > 0.30:
                    st.success(f"Strong visual match ({raw_score:.2f})")
                elif raw_score > threshold:
                    st.info(f"Moderate visual match ({raw_score:.2f})")
                else:
                    st.warning(f"Weak visual similarity ({raw_score:.2f})")

                st.caption(f"CLIP cosine similarity: {raw_score:.4f}")

        else:
            st.warning("No visual match found.")
