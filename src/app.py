import streamlit as st
import requests
from PIL import Image
import io
import base64
import time
from serpapi import GoogleSearch # NEW LIBRARY

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8000"
# 👇 PASTE YOUR SERPAPI KEY HERE (Get it from serpapi.com)
SERPAPI_KEY = "0d3e917f0ddb66d5d989c8be3ae84302f3f819596119589eb279331be6f55e8e" 

st.set_page_config(page_title="Enterprise Visual Search", page_icon="🏢", layout="wide")

def base64_to_image(b64_str):
    img_data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_data))

# --- SESSION STATE INITIALIZATION ---
if "cam_on" not in st.session_state: st.session_state.cam_on = False
if "uploader_key" not in st.session_state: st.session_state.uploader_key = 0
if "preview_buffer" not in st.session_state: st.session_state.preview_buffer = []

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ System Controls")
    
    try:
        status = requests.get(f"{API_URL}/health", timeout=5).json()
        st.success(f"🟢 API Online | Indexed: {status['vectors_indexed']}")
    except:
        st.error("🔴 API Offline. Run 'python src/api.py'")
        st.stop()

    # st.info("🧠 Auto-Captioning Active.")
    st.divider()

    st.subheader("📥 Data Ingestion")
    ingest_mode = st.radio("Source:", ["📂 Manual Upload", "🌐 Web Crawler (Google)"], horizontal=True)

    # --- METHOD 1: MANUAL UPLOAD ---
    if ingest_mode == "📂 Manual Upload":
        files = st.file_uploader(
            "Drag & Drop Files", 
            accept_multiple_files=True,
            type=['png', 'jpg', 'jpeg', 'bmp', 'webp', 'tiff', 'tif', 'jfif'],
            key=f"uploader_{st.session_state.uploader_key}"
        )
        
        if files and st.button("Transmit to Server", type="primary"):
            payload = [('files', (f.name, f, f.type)) for f in files]
            with st.spinner("Transmitting..."):
                try:
                    res = requests.post(f"{API_URL}/ingest", files=payload)
                    if res.status_code == 200:
                        st.toast("Ingestion Complete!", icon="✅")
                        st.session_state.uploader_key += 1 
                        st.rerun()
                    else:
                        st.error("Server failed.")
                except Exception as e:
                    st.error(f"Error: {e}")

    # --- METHOD 2: GOOGLE SEARCH (SERPAPI) ---
    elif ingest_mode == "🌐 Web Crawler (Google)":
        
        with st.form("web_search_form"):
            topic = st.text_input("Topic", placeholder="e.g. Red Sports Car")
            count = st.number_input("Count", min_value=1, max_value=20, value=5, step=1)
            search_submitted = st.form_submit_button("🔍 Search Google", type="primary")
        
        if search_submitted and topic:
            if "PASTE_YOUR_KEY" in SERPAPI_KEY:
                st.error("⚠️ You forgot to paste your SerpApi Key in the code!")
                st.stop()

            st.session_state.preview_buffer = [] 
            status_text = st.empty()
            
            # 1. SEARCH GOOGLE (Robust & Block-Free)
            status_text.write(f"🔍 Searching Google Images for '{topic}'...")
            
            params = {
                "q": topic,
                "tbm": "isch", # 'isch' = Image Search
                "ijn": "0",
                "api_key": "0d3e917f0ddb66d5d989c8be3ae84302f3f819596119589eb279331be6f55e8e"
            }
            
            try:
                search = GoogleSearch(params)
                results = search.get_dict()
                images_results = results.get("images_results", [])
                
                # Get the URLs
                image_urls = [img['original'] for img in images_results][:count]
                
            except Exception as e:
                st.error(f"Google Search Error: {e}")
                st.stop()

            # 2. DOWNLOAD PREVIEWS
            downloaded = 0
            if image_urls:
                progress = st.progress(0)
                for i, url in enumerate(image_urls):
                    try:
                        resp = requests.get(url, timeout=3)
                        if resp.status_code == 200:
                            img_obj = Image.open(io.BytesIO(resp.content))
                            st.session_state.preview_buffer.append({
                                "id": downloaded,
                                "bytes": resp.content,
                                "img_obj": img_obj,
                                "topic": topic,
                                "selected": True 
                            })
                            downloaded += 1
                    except:
                        pass
                    progress.progress(min((i + 1) / len(image_urls), 1.0))
                
                status_text.success(f"✅ Found {downloaded} images. Review below!")
            else:
                status_text.warning("No images found.")

        # 3. REVIEW & UPLOAD
        if st.session_state.preview_buffer:
            st.divider()
            st.write("### 👁️ Review Google Results")
            st.caption("Uncheck images to discard.")
            
            with st.form("review_upload_form"):
                cols = st.columns(2)
                selected_indices = []
                
                for idx, item in enumerate(st.session_state.preview_buffer):
                    col = cols[idx % 2]
                    with col:
                        st.image(item["img_obj"], use_container_width=True)
                        if st.checkbox("Keep", value=True, key=f"check_{item['id']}"):
                            selected_indices.append(idx)
                
                st.divider()
                upload_clicked = st.form_submit_button("🚀 Upload to Enterprise DB")

            if upload_clicked:
                if not selected_indices:
                    st.warning("No images selected!")
                else:
                    success_count = 0
                    with st.status("Uploading...", expanded=True) as status:
                        for idx in selected_indices:
                            item = st.session_state.preview_buffer[idx]
                            filename = f"google_{item['topic'].replace(' ', '_')}_{item['id']}.jpg"
                            files_payload = {'files': (filename, item['bytes'], 'image/jpeg')}
                            try:
                                res = requests.post(f"{API_URL}/ingest", files=files_payload)
                                if res.status_code == 200:
                                    success_count += 1
                                    status.write(f"✅ Uploaded {filename}")
                            except Exception as e:
                                status.write(f"❌ Failed {filename}: {e}")
                        status.update(label=f"Done! Uploaded {success_count} images.", state="complete")
                    st.session_state.preview_buffer = []
                    st.button("Reset")

# --- MAIN INTERFACE ---
st.title("Visual Recognition")

tab1, tab2 = st.tabs(["📷 Live Feed", "📂 File Analysis"])
query_img = None

with tab1:
    c1, c2, _ = st.columns([1, 1, 3])
    with c1:
        if st.button("Start camera", type="primary", use_container_width=True):
            st.session_state.cam_on = True
            st.rerun()
    with c2:
        if st.button("Stop camera", use_container_width=True):
            st.session_state.cam_on = False
            st.rerun()
            
    if st.session_state.cam_on:
        cam = st.camera_input("Feed", label_visibility="collapsed")
        if cam: query_img = Image.open(cam)

with tab2:
    up = st.file_uploader("Upload Query Image", type=['png', 'jpg', 'jpeg', 'bmp', 'webp', 'tiff', 'tif','jfif'])
    if up: query_img = Image.open(up)

if query_img:
    st.divider()
    buf = io.BytesIO()
    query_img.save(buf, format="PNG")
    api_files = {'file': ('query.png', buf.getvalue(), 'image/png')}
    
    with st.spinner("Analyzing Scene..."):
        try:
            response = requests.post(f"{API_URL}/predict", files=api_files).json()
        except Exception as e:
            st.error(f"API Error: {e}")
            st.stop()
            
    caption = response['caption']
    visual = response['visual_match']
    
    c_in, c_out = st.columns([1, 2])
    with c_in:
        st.write("#### Input Source")
        st.image(query_img, width=300)
        
    with c_out:
        st.write("#### Intelligence Report")
        st.info(f"**AI Description:** {caption}")
        st.write("---")
        
        if visual:
            raw_score = visual['score']
            threshold = 0.40
            if raw_score < threshold:
                st.warning(f"No match found. (Closest: {raw_score:.2f})")
            else:
                display_conf = ((raw_score - threshold) / (1 - threshold)) * 100
                st.write("**Database Match Found:**")
                vc1, vc2 = st.columns([1, 2])
                with vc1: st.image(base64_to_image(visual['base64']), width=150)
                with vc2:
                    if display_conf > 80: st.success(f"Exact Match ({display_conf:.1f}%)")
                    elif display_conf > 60: st.info(f"Likely Match ({display_conf:.1f}%)")
                    else: st.warning(f"Low Confidence ({display_conf:.1f}%)")
        else:
            st.warning("No visual match found.")