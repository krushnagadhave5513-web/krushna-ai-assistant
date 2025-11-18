# KrushnaAI_full_upgrade.py
# Krushna AI Assistant — Upgraded (chat, image/video gen, analysis, sessions, UI)
# Requirements:
# streamlit openai pillow requests PyPDF2 python-docx imageio moviepy (moviepy optional)

import streamlit as st
from openai import OpenAI
from PIL import Image, ImageOps, ExifTags
import io, os, time, json, base64, tempfile, traceback, requests
import threading
import textwrap

# Optional packages (best-effort)
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    import docx
except Exception:
    docx = None

try:
    import imageio
except Exception:
    imageio = None

try:
    from moviepy.editor import ImageSequenceClip
except Exception:
    ImageSequenceClip = None

# ------------------------
# Utility functions
# ------------------------
def safe_print(*a, **k):
    try:
        print(*a, **k)
    except Exception:
        pass

def sanitize_messages(messages):
    out = []
    if not messages:
        return []
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    for m in messages:
        try:
            if isinstance(m, str):
                out.append({"role": "user", "content": m})
            elif not isinstance(m, dict):
                out.append({"role": "user", "content": str(m)})
            else:
                role = m.get("role", "user") or "user"
                content = m.get("content", "") or ""
                out.append({"role": role, "content": content})
        except Exception:
            out.append({"role": "user", "content": str(m)})
    return out

def extract_assistant_text_from_raw(raw):
    """Robust extractor for OpenAI responses (modern and legacy)."""
    try:
        # object-like
        if hasattr(raw, "choices") and len(raw.choices) > 0:
            c0 = raw.choices[0]
            try:
                if hasattr(c0, "message") and c0.message:
                    msg = c0.message
                    if hasattr(msg, "content"):
                        return msg.content or ""
                    if hasattr(msg, "get"):
                        return msg.get("content", "") or ""
            except Exception:
                pass
            # delta streaming
            try:
                if hasattr(c0, "delta") and c0.delta:
                    d = c0.delta
                    if hasattr(d, "content"):
                        return d.content or ""
                    if hasattr(d, "get"):
                        return d.get("content", "") or ""
            except Exception:
                pass
            if hasattr(c0, "text"):
                return getattr(c0, "text") or ""
        # dict-like
        if isinstance(raw, dict):
            choices = raw.get("choices", [])
            if choices:
                c0 = choices[0]
                if isinstance(c0, dict):
                    msg = c0.get("message") or c0.get("delta") or {}
                    if isinstance(msg, dict):
                        return msg.get("content") or msg.get("text") or ""
                    return c0.get("text", "") or ""
        return str(raw)
    except Exception:
        try:
            return str(raw)
        except Exception:
            return "(Could not parse response)"

def pil_fix_alpha_make_white(im: Image.Image):
    try:
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            bg = Image.new("RGBA", im.size, (255,255,255,255))
            bg.paste(im, mask=im.split()[-1])
            return bg.convert("RGB")
        else:
            return im.convert("RGB")
    except Exception:
        return im

def image_bytes_from_base64(b64str):
    return base64.b64decode(b64str)

def create_gif_from_bytes_list(image_bytes_list, out_path, duration=0.5):
    """Create GIF using imageio if available, else use PIL."""
    try:
        if imageio:
            frames = [imageio.imread(io.BytesIO(b)) for b in image_bytes_list]
            imageio.mimsave(out_path, frames, duration=duration)
            return out_path
        else:
            # fallback PIL
            frames = [Image.open(io.BytesIO(b)).convert("RGBA") for b in image_bytes_list]
            frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=int(duration*1000), loop=0)
            return out_path
    except Exception as e:
        safe_print("GIF creation failed:", e)
        return None

def create_mp4_from_bytes_list(image_bytes_list, out_path, fps=4):
    try:
        # requires moviepy
        if ImageSequenceClip is None:
            return None
        tmp_dir = tempfile.mkdtemp()
        file_paths = []
        for i, b in enumerate(image_bytes_list):
            p = os.path.join(tmp_dir, f"frame_{i:03d}.png")
            with open(p, "wb") as f:
                f.write(b)
            file_paths.append(p)
        clip = ImageSequenceClip(file_paths, fps=fps)
        clip.write_videofile(out_path, codec="libx264", audio=False, verbose=False, logger=None)
        return out_path
    except Exception as e:
        safe_print("MP4 creation failed:", e)
        return None

def extract_text_from_pdf_bytes(pdf_bytes, max_pages=3):
    if PyPDF2 is None:
        return "(PyPDF2 not installed on server)"
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for p in reader.pages[:max_pages]:
            try:
                text += p.extract_text() or ""
            except Exception:
                pass
        return text
    except Exception as e:
        return f"(PDF parse error: {e})"

def extract_text_from_docx_bytes(docx_bytes):
    if docx is None:
        return "(python-docx not installed on server)"
    try:
        tmp = io.BytesIO(docx_bytes)
        doc = docx.Document(tmp)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs[:2000])
    except Exception as e:
        return f"(DOCX parse error: {e})"

def analyze_image_bytes_basic(img_bytes):
    """Return small metadata and a short textual descriptor used for analysis."""
    try:
        im = Image.open(io.BytesIO(img_bytes))
        w,h = im.size
        mode = im.mode
        # attempt to read EXIF
        exif = {}
        try:
            raw_exif = getattr(im, "_getexif", lambda: None)()
            if raw_exif:
                for k,v in raw_exif.items():
                    tag = ExifTags.TAGS.get(k,k)
                    exif[tag] = v
        except Exception:
            exif = {}
        desc = f"Image ({w}x{h}, mode={mode}). EXIF keys: {list(exif.keys())[:6]}"
        return desc
    except Exception as e:
        return f"(Image analysis failed: {e})"

# ------------------------
# Streamlit UI + state
# ------------------------
st.set_page_config(page_title="Krushna AI Assistant", page_icon="🤖", layout="wide")
st.title("🤖 Krushna AI Assistant — Upgraded")
st.write("Image & Video generation, analysis, multilingual, sessions, improved UI.")

# ensure API_KEY secret present
if "API_KEY" not in st.secrets:
    st.error("Streamlit secret `API_KEY` missing. Add it in app settings.")
    st.stop()

API_KEY = st.secrets["API_KEY"]

# initialize client once
if "client" not in st.session_state:
    try:
        st.session_state.client = OpenAI(api_key=API_KEY)
    except Exception as e:
        st.error("OpenAI client init failed: " + str(e))
        st.stop()

client = st.session_state.client

# session variables
if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"system","content":"You are Jarvis 2.0, helpful assistant for Krushna. Keep answers concise and code-friendly."}]
if "saved_sessions" not in st.session_state:
    st.session_state.saved_sessions = {}  # name -> messages
if "images" not in st.session_state:
    st.session_state.images = []  # bytes list
if "status" not in st.session_state:
    st.session_state.status = "Ready"
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = st.session_state.messages[0]["content"]

# ------------------------
# Sidebar (left): sessions, controls, analysis options
# ------------------------
with st.sidebar:
    st.header("Sessions & Controls")

    # custom system prompt
    st.subheader("System prompt")
    sp = st.text_area("Control Jarvis behavior (system prompt)", value=st.session_state.system_prompt, height=120)
    if st.button("Apply system prompt"):
        st.session_state.system_prompt = sp.strip()
        # replace first system message
        if len(st.session_state.messages)>0 and st.session_state.messages[0].get("role")=="system":
            st.session_state.messages[0]["content"] = st.session_state.system_prompt
        else:
            st.session_state.messages.insert(0, {"role":"system","content":st.session_state.system_prompt})
        st.success("System prompt applied")

    st.markdown("---")
    st.subheader("Saved sessions")
    # show saved sessions with load/delete/rename
    if st.session_state.saved_sessions:
        for name in list(st.session_state.saved_sessions.keys()):
            cols = st.columns([4,1,1])
            cols[0].write(name)
            if cols[1].button("Load", key=f"load_{name}"):
                st.session_state.messages = sanitize_messages(st.session_state.saved_sessions[name])
                st.success(f"Loaded session: {name}")
            if cols[2].button("Delete", key=f"del_{name}"):
                del st.session_state.saved_sessions[name]
                st.success(f"Deleted session: {name}")
                st.experimental_rerun()
    else:
        st.info("No saved sessions yet")

    st.text_input("New session name", key="new_session_name")
    if st.button("Save current session"):
        nm = st.session_state.get("new_session_name") or f"session_{time.strftime('%Y%m%d_%H%M%S')}"
        st.session_state.saved_sessions[nm] = list(st.session_state.messages)
        st.success(f"Saved session: {nm}")

    if st.button("Export current (.txt)"):
        lines = []
        for m in st.session_state.messages:
            role = m.get("role","user")
            prefix = "You: " if role=="user" else ("Jarvis: " if role=="assistant" else f"{role}: ")
            lines.append(prefix + m.get("content",""))
        txt = "\n\n".join(lines)
        st.download_button("Download transcript", txt, file_name=f"transcript_{int(time.time())}.txt", mime="text/plain")

    st.markdown("---")
    st.subheader("Analysis / Tools")
    st.write("Upload files below for analysis (PDF/DOCX/Image/Video).")
    up_files = st.file_uploader("Upload files (multiple)", accept_multiple_files=True)
    st.markdown("**Image generation**")
    gen_prompt = st.text_input("Image prompt (sidebar)", key="gen_prompt_sidebar")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Generate 1"):
            st.session_state._gen_request = {"prompt":gen_prompt,"n":1}
    with col2:
        if st.button("Generate 4"):
            st.session_state._gen_request = {"prompt":gen_prompt,"n":4}

    st.markdown("---")
    st.subheader("Language & Speed")
    lang_choice = st.selectbox("Respond in language (leave default for same language)", ["(same)", "English", "Hindi", "Marathi", "Spanish", "French", "Auto-detect"])
    speed_mode = st.selectbox("Speed Mode", ["Normal", "Faster (low latency)"])
    st.write("Status:", st.session_state.status)

# ------------------------
# Main UI layout: left chat, right preview/analysis
# ------------------------
left_col, right_col = st.columns([2.5,1])

with left_col:
    st.subheader("Conversation")
    # display messages as chat bubbles
    for m in st.session_state.messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        try:
            with st.chat_message(role):
                # formatted block for long text
                st.markdown(content)
        except Exception:
            if role=="user":
                st.markdown(f"**You:** {content}")
            elif role=="assistant":
                st.markdown(f"**Jarvis:** {content}")
            else:
                st.markdown(f"**{role}:** {content}")

    # Input: Enter sends immediately (st.chat_input)
    user_input = st.chat_input("Type message (press Enter to send)...")
    if user_input is not None and user_input.strip() != "":
        # append user message
        st.session_state.messages.append({"role":"user","content":user_input})
        # store in memory optionally
        # (we include every user message in memory list; user can clear later)
        if "memory_list" not in st.session_state:
            st.session_state.memory_list = []
        st.session_state.memory_list.append(user_input)

        # show typing animation & call model
        st.session_state.status = "Jarvis is thinking..."
        # typing placeholder
        placeholder = st.empty()
        with st.spinner("Jarvis is thinking..."):
            try:
                # prepare messages copy and apply language/system prompt if changed
                messages_for_api = sanitize_messages(st.session_state.messages)
                # speed presets
                if speed_mode == "Faster (low latency)":
                    temperature = 0.2
                    max_tokens = 400
                else:
                    temperature = 0.5
                    max_tokens = 800

                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages_for_api,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                assistant_text = extract_assistant_text_from_raw(resp)
                if not isinstance(assistant_text, str):
                    assistant_text = str(assistant_text)
                # if language choice forced, ask model to respond in that language:
                if lang_choice and lang_choice != "(same)" and lang_choice != "Auto-detect":
                    # Ask model to translate/answer in the chosen language
                    assistant_text = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role":"system","content":f"Respond in {lang_choice}."},
                            {"role":"user","content":assistant_text}
                        ]
                    )
                    assistant_text = extract_assistant_text_from_raw(assistant_text)
                    if not isinstance(assistant_text, str):
                        assistant_text = str(assistant_text)

                # typing animation: reveal in chunks
                placeholder.markdown("Jarvis is typing...")
                # chunk and display progressively
                display_box = st.empty()
                words = assistant_text.split()
                chunk_size = 8
                built = ""
                for i in range(0, len(words), chunk_size):
                    built += " " + " ".join(words[i:i+chunk_size])
                    display_box.markdown(built + "▌")
                    time.sleep(0.04)  # small delay for typing effect
                display_box.markdown(built)  # final

                # finalize
                st.session_state.messages.append({"role":"assistant","content":assistant_text})
                st.session_state.status = "Ready"
                placeholder.empty()
            except Exception as e:
                tb = traceback.format_exc()
                st.session_state.messages.append({"role":"assistant","content":f"API Error: {e}\n\n{tb}"})
                st.session_state.status = "Error"
                placeholder.empty()
                st.error("API Error: " + str(e))

with right_col:
    st.subheader("Images / Analysis / Video")
    # show generated/preview images
    if st.session_state.images:
        st.write(f"Preview ({len(st.session_state.images)} images)")
        for i,b in enumerate(st.session_state.images):
            try:
                st.image(b, caption=f"Generated {i+1}", use_column_width=True)
            except Exception:
                st.write("Could not preview image.")
        # download buttons
        for i,b in enumerate(st.session_state.images):
            st.download_button(f"Download image {i+1}", data=b, file_name=f"gen_{i+1}.png", mime="image/png")

        # allow make GIF / MP4
        if st.button("Create GIF from generated images"):
            tmp_gif = os.path.join(tempfile.gettempdir(), f"krushna_gen_{int(time.time())}.gif")
            path = create_gif_from_bytes_list(st.session_state.images, tmp_gif, duration=0.6)
            if path:
                with open(path, "rb") as f:
                    st.download_button("Download GIF", f, file_name=os.path.basename(path), mime="image/gif")
            else:
                st.error("GIF creation failed.")

        if st.button("Create MP4 from generated images (requires moviepy)"):
            tmp_mp4 = os.path.join(tempfile.gettempdir(), f"krushna_gen_{int(time.time())}.mp4")
            path = create_mp4_from_bytes_list(st.session_state.images, tmp_mp4, fps=4)
            if path:
                with open(path, "rb") as f:
                    st.download_button("Download MP4", f, file_name=os.path.basename(path), mime="video/mp4")
            else:
                st.error("MP4 creation failed (moviepy not available or error).")

    else:
        st.info("No generated images yet - use sidebar or quick generator below.")

    st.markdown("---")
    st.subheader("Quick generator (right panel)")
    q_prompt = st.text_input("Quick prompt (right)", key="quick_right_prompt")
    colA, colB = st.columns(2)
    with colA:
        if st.button("Gen 1 (right)"):
            p = st.session_state.get("quick_right_prompt","").strip()
            if not p:
                st.warning("Enter prompt first.")
            else:
                st.session_state.status = "Generating images..."
                try:
                    # attempt modern images API
                    images_info = []
                    try:
                        resp = client.images.generate(model="gpt-image-1", prompt=p, size="1024x1024", n=1)
                        data_list = getattr(resp, "data", None) or resp.get("data", [])
                        for d in data_list:
                            if hasattr(d, "b64_json"):
                                images_info.append({"b64": d.b64_json})
                            elif isinstance(d, dict) and d.get("b64_json"):
                                images_info.append({"b64": d.get("b64_json")})
                            elif hasattr(d, "url"):
                                images_info.append({"url": d.url})
                            elif isinstance(d, dict) and d.get("url"):
                                images_info.append({"url": d.get("url")})
                    except Exception:
                        images_info = []
                    # fallback legacy
                    if not images_info:
                        try:
                            resp = client.Image.create(prompt=p, n=1, size="1024x1024")
                            for d in resp.get("data", []):
                                images_info.append({"b64": d.get("b64_json"), "url": d.get("url")})
                        except Exception:
                            pass
                    saved = []
                    for info in images_info:
                        img_bytes = None
                        if info.get("url"):
                            try:
                                r = requests.get(info["url"], timeout=15)
                                r.raise_for_status()
                                img_bytes = r.content
                            except Exception:
                                b64 = info.get("b64")
                                if b64:
                                    img_bytes = base64.b64decode(b64)
                        elif info.get("b64"):
                            img_bytes = base64.b64decode(info.get("b64"))
                        if img_bytes:
                            try:
                                im = Image.open(io.BytesIO(img_bytes))
                                fixed = pil_fix_alpha_make_white(im)
                                buff = io.BytesIO()
                                fixed.save(buff, format="PNG")
                                st.session_state.images.append(buff.getvalue())
                                saved.append(buff.getvalue())
                            except Exception:
                                pass
                    if saved:
                        st.success(f"Generated {len(saved)} image(s).")
                    else:
                        st.warning("No images returned.")
                except Exception as e:
                    st.error("Image generation error: " + str(e))
                finally:
                    st.session_state.status = "Ready"

    with colB:
        if st.button("Gen 4 (right)"):
            p = st.session_state.get("quick_right_prompt","").strip()
            if not p:
                st.warning("Enter prompt first.")
            else:
                st.session_state.status = "Generating images..."
                try:
                    images_info = []
                    try:
                        resp = client.images.generate(model="gpt-image-1", prompt=p, size="1024x1024", n=4)
                        data_list = getattr(resp, "data", None) or resp.get("data", [])
                        for d in data_list:
                            if hasattr(d, "b64_json"):
                                images_info.append({"b64": d.b64_json})
                            elif isinstance(d, dict) and d.get("b64_json"):
                                images_info.append({"b64": d.get("b64_json")})
                            elif hasattr(d, "url"):
                                images_info.append({"url": d.url})
                            elif isinstance(d, dict) and d.get("url"):
                                images_info.append({"url": d.get("url")})
                    except Exception:
                        images_info = []
                    if not images_info:
                        try:
                            resp = client.Image.create(prompt=p, n=4, size="1024x1024")
                            for d in resp.get("data", []):
                                images_info.append({"b64": d.get("b64_json"), "url": d.get("url")})
                        except Exception:
                            pass
                    saved = []
                    for info in images_info:
                        img_bytes = None
                        if info.get("url"):
                            try:
                                r = requests.get(info["url"], timeout=15)
                                r.raise_for_status()
                                img_bytes = r.content
                            except Exception:
                                b64 = info.get("b64")
                                if b64:
                                    img_bytes = base64.b64decode(b64)
                        elif info.get("b64"):
                            img_bytes = base64.b64decode(info.get("b64"))
                        if img_bytes:
                            try:
                                im = Image.open(io.BytesIO(img_bytes))
                                fixed = pil_fix_alpha_make_white(im)
                                buff = io.BytesIO()
                                fixed.save(buff, format="PNG")
                                st.session_state.images.append(buff.getvalue())
                                saved.append(buff.getvalue())
                            except Exception:
                                pass
                    if saved:
                        st.success(f"Generated {len(saved)} images.")
                    else:
                        st.warning("No images returned.")
                except Exception as e:
                    st.error("Image generation error: " + str(e))
                finally:
                    st.session_state.status = "Ready"

    st.markdown("---")
    st.subheader("File analysis (uploaded files from sidebar)")

    # handle uploaded files (from sidebar)
    try:
        uploaded_files = up_files  # variable captured earlier in sidebar
    except Exception:
        uploaded_files = None

    if uploaded_files:
        for f in uploaded_files:
            fname = f.name
            st.write(f"**Processing** {fname}")
            data = f.read()
            if fname.lower().endswith((".png",".jpg",".jpeg")):
                desc = analyze_image_bytes_basic(data)
                st.write("Image analysis:", desc)
                # send to model for deeper analysis
                try:
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role":"system","content":st.session_state.system_prompt},
                            {"role":"user","content":f"Analyze this image and give a short descriptive summary and notable features. Metadata: {desc}"}
                        ]
                    )
                    analysis = extract_assistant_text_from_raw(resp)
                    st.write("Jarvis analysis:", analysis)
                    st.session_state.messages.append({"role":"assistant","content":f"[Image Analysis] {analysis}"})
                except Exception as e:
                    st.error("Image analysis API error: " + str(e))

            elif fname.lower().endswith(".pdf"):
                txt = extract_text_from_pdf_bytes(data)
                st.write("Extracted PDF text (snippet):", txt[:2000])
                try:
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role":"system","content":st.session_state.system_prompt},
                            {"role":"user","content":f"Summarize and extract key points from this PDF text:\n\n{txt[:5000]}"}
                        ]
                    )
                    analysis = extract_assistant_text_from_raw(resp)
                    st.write("Jarvis analysis:", analysis)
                    st.session_state.messages.append({"role":"assistant","content":f"[PDF Analysis] {analysis}"})
                except Exception as e:
                    st.error("PDF analysis API error: " + str(e))

            elif fname.lower().endswith(".docx") or fname.lower().endswith(".doc"):
                txt = extract_text_from_docx_bytes(data)
                st.write("Extracted DOCX text (snippet):", txt[:2000])
                try:
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role":"system","content":st.session_state.system_prompt},
                            {"role":"user","content":f"Summarize and extract key points from this document:\n\n{txt[:5000]}"}
                        ]
                    )
                    analysis = extract_assistant_text_from_raw(resp)
                    st.write("Jarvis analysis:", analysis)
                    st.session_state.messages.append({"role":"assistant","content":f"[DOCX Analysis] {analysis}"})
                except Exception as e:
                    st.error("DOCX analysis API error: " + str(e))
            else:
                # other file types: show size
                st.write("Uploaded file type not directly supported for deep analysis. Size:", len(data), "bytes")

# ------------------------
# Small footer controls
# ------------------------
st.markdown("---")
colL, colR = st.columns([3,1])
with colL:
    st.write("Features included: image gen, video (GIF/MP4) generation, file analysis (PDF/DOCX/Image), multilingual reply, sessions, custom system prompt, speed mode, typing animation, multiple image upload.")
with colR:
    if st.button("Reset all (clear sessions & chat)"):
        st.session_state.messages = [{"role":"system","content":st.session_state.system_prompt}]
        st.session_state.saved_sessions = {}
        st.session_state.images = []
        st.session_state.status = "Ready"
        st.success("Reset done.")

# End of file

