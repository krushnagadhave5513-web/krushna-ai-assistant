# KrushnaAI.py
# Krushna AI Assistant (Streamlit single-file app)
# Features:
# - Chat (chat history preserved in session)
# - Memory / sessions (save/load/export)
# - Image generation (1 or 4)
# - PDF & Image upload (PDF text extract if PyPDF2 installed)
# - Fix alpha PNG (avoid black background)
# - Defensive parsing of OpenAI responses (new + legacy shapes)
# - Streamlit-compatible (no experimental_rerun)
#
# Requirements:
# pip install streamlit openai pillow requests PyPDF2

import streamlit as st
from openai import OpenAI
from PIL import Image
import io, os, json, time, base64, tempfile, traceback, requests

# Optional PDF support
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

# -------------------- Helpers --------------------

def sanitize_messages(messages):
    """Return a safe list of message dicts {'role','content'}."""
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
                role = m.get("role") or "user"
                content = m.get("content") or ""
                out.append({"role": role, "content": content})
        except Exception:
            out.append({"role": "user", "content": str(m)})
    return out

def extract_assistant_text_from_raw(raw):
    """Robust extractor for many shapes of OpenAI SDK responses."""
    try:
        # object-like typical modern SDK
        if hasattr(raw, "choices") and len(raw.choices) > 0:
            choice0 = raw.choices[0]
            try:
                if hasattr(choice0, "message") and choice0.message:
                    msg = choice0.message
                    # attribute .content
                    if hasattr(msg, "content"):
                        return msg.content or ""
                    # dict-like .get
                    if hasattr(msg, "get"):
                        return msg.get("content", "") or ""
            except Exception:
                pass
            # delta
            try:
                if hasattr(choice0, "delta") and choice0.delta:
                    delta = choice0.delta
                    if hasattr(delta, "content"):
                        return delta.content or ""
                    if hasattr(delta, "get"):
                        return delta.get("content", "") or ""
            except Exception:
                pass
            # fallback 'text'
            if hasattr(choice0, "text"):
                return getattr(choice0, "text") or ""
        # dict-like fallback (legacy)
        if isinstance(raw, dict):
            choices = raw.get("choices", [])
            if choices:
                c0 = choices[0]
                if isinstance(c0, dict):
                    # try message -> content
                    msg = c0.get("message") or c0.get("delta") or {}
                    if isinstance(msg, dict):
                        return msg.get("content") or msg.get("text") or ""
                    return c0.get("text", "") or ""
        # final fallback
        return str(raw)
    except Exception:
        try:
            return str(raw)
        except Exception:
            return "(Could not parse assistant response)"

def pil_fix_alpha_make_white(im: Image.Image):
    """If image has alpha channel, put it over white background and return RGB bytes."""
    try:
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            out = bg.convert("RGB")
        else:
            out = im.convert("RGB")
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        # fallback: try convert
        try:
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="PNG")
            return buf.getvalue()
        except Exception:
            return None

def save_temp_json(name_prefix, data):
    try:
        fname = f"{name_prefix}_{int(time.time())}.json"
        path = os.path.join(tempfile.gettempdir(), fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path
    except Exception:
        return None

# -------------------- Streamlit UI Setup --------------------

st.set_page_config(page_title="Krushna AI Assistant", page_icon="🤖", layout="wide")
st.title("🤖 Krushna AI Assistant")
st.write("Chat, generate images, upload files, save sessions — powered by OpenAI.")

# -------------------- Load API Key --------------------
if "API_KEY" not in st.secrets:
    st.error("Missing Streamlit secret `API_KEY`. Add it in the app settings and redeploy.")
    st.stop()

API_KEY = st.secrets["API_KEY"]

# Initialize OpenAI client once
if "client" not in st.session_state:
    try:
        st.session_state.client = OpenAI(api_key=API_KEY)
    except Exception as e:
        st.error("Could not initialize OpenAI client: " + str(e))
        st.stop()

client = st.session_state.client

# -------------------- Session State Defaults --------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are Jarvis 2.0, helpful assistant for Krushna. Keep answers concise and code-friendly."}
    ]
if "saved_sessions" not in st.session_state:
    st.session_state.saved_sessions = {}  # name -> messages
if "images" not in st.session_state:
    st.session_state.images = []  # list of bytes
if "status" not in st.session_state:
    st.session_state.status = "Ready"

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("Sessions & Tools")

    # Saved sessions picker
    saved_names = list(st.session_state.saved_sessions.keys())
    if saved_names:
        sel = st.selectbox("Load saved session", options=["-- choose --"] + saved_names)
        if sel and sel != "-- choose --":
            if st.button("Load selected session"):
                st.session_state.messages = sanitize_messages(st.session_state.saved_sessions.get(sel, []))
                st.success(f"Loaded session: {sel}")
    else:
        st.info("No saved sessions yet")

    if st.button("Save current session"):
        name = f"session_{time.strftime('%Y%m%d_%H%M%S')}"
        st.session_state.saved_sessions[name] = list(st.session_state.messages)
        st.success(f"Saved session: {name}")

    if st.button("Export current as .txt"):
        lines = []
        for m in st.session_state.messages:
            role = m.get("role", "user")
            prefix = "You: " if role == "user" else ("Jarvis: " if role == "assistant" else f"{role.capitalize()}: ")
            lines.append(prefix + m.get("content", ""))
        transcript = "\n\n".join(lines)
        st.download_button("Download transcript (.txt)", transcript, file_name=f"transcript_{int(time.time())}.txt", mime="text/plain")

    st.markdown("---")
    st.header("Upload")
    uploaded_image = st.file_uploader("Upload image (png/jpg)", type=["png", "jpg", "jpeg"])
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
    st.markdown("---")
    st.header("Image Generation")
    image_prompt = st.text_input("Image prompt", key="image_prompt_input")
    cols = st.columns([1,1,1])
    with cols[0]:
        gen1 = st.button("Generate 1 image")
    with cols[1]:
        gen4 = st.button("Generate 4 images")
    with cols[2]:
        clr_imgs = st.button("Clear previews")

    st.markdown("---")
    st.write("Status:")
    st.info(st.session_state.status)

# -------------------- Main layout --------------------
chat_col, preview_col = st.columns([2.5, 1])

with chat_col:
    st.subheader("Conversation")
    # Render chat history
    for m in st.session_state.messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        # Use streamlit's chat_message if available
        try:
            with st.chat_message(role):
                st.markdown(content)
        except Exception:
            if role == "user":
                st.markdown(f"**You:** {content}")
            elif role == "assistant":
                st.markdown(f"**Jarvis:** {content}")
            else:
                st.markdown(f"**{role}:** {content}")

    # Input area
    user_input = st.text_input("Type your message:", key="main_input")
    send_btn = st.button("Send")

    # Quick actions
    q1, q2, q3 = st.columns([1,1,1])
    with q1:
        if st.button("Send last"):
            last_user = None
            for mm in reversed(st.session_state.messages):
                if mm.get("role") == "user":
                    last_user = mm.get("content")
                    break
            if last_user:
                st.session_state.main_input = last_user
                # let Send process below
                send_btn = True
    with q2:
        if st.button("Clear chat"):
            st.session_state.messages = [m for m in st.session_state.messages if m.get("role") == "system"]
            st.success("Cleared chat (system message retained).")
    with q3:
        if st.button("Save snapshot (.json)"):
            path = save_temp_json("snapshot", st.session_state.messages)
            if path:
                with open(path, "rb") as f:
                    st.download_button("Download snapshot", f, file_name=os.path.basename(path))
            else:
                st.error("Could not save snapshot.")

    # Process uploaded image
    if uploaded_image is not None:
        try:
            img = Image.open(uploaded_image)
            img_bytes = pil_fix_alpha_make_white(img)
            st.session_state.images.append(img_bytes)
            st.session_state.messages.append({"role": "user", "content": f"[Uploaded Image] {uploaded_image.name}"})
            st.success(f"Image uploaded: {uploaded_image.name}")
        except Exception as e:
            st.error("Image upload error: " + str(e))

    # Process uploaded PDF
    if uploaded_pdf is not None:
        st.session_state.messages.append({"role": "user", "content": f"[Uploaded PDF] {uploaded_pdf.name}"})
        if PyPDF2 is None:
            st.warning("PyPDF2 not installed; PDF text extraction unavailable.")
        else:
            try:
                reader = PyPDF2.PdfReader(uploaded_pdf)
                text = ""
                for p in reader.pages[:3]:
                    try:
                        text += p.extract_text() or ""
                    except Exception:
                        pass
                snippet = (text[:1000] + "...") if len(text) > 1000 else text
                if snippet.strip():
                    st.session_state.messages.append({"role": "assistant", "content": f"[PDF Extract]\n{snippet}"})
                    st.success("PDF text extracted (first pages).")
                else:
                    st.session_state.messages.append({"role": "assistant", "content": "[PDF] Could not extract text."})
            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"[PDF] Error: {e}"})

    # SEND flow (synchronous, shows spinner)
    if send_btn and st.session_state.get("main_input", "").strip():
        text = st.session_state.get("main_input", "").strip()
        # Append user message
        st.session_state.messages.append({"role": "user", "content": text})
        st.session_state.status = "Jarvis is thinking..."
        # Call OpenAI synchronously with spinner
        with st.spinner("Jarvis is thinking..."):
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=st.session_state.messages,
                    max_tokens=800,
                    temperature=0.4
                )
                assistant_text = extract_assistant_text_from_raw(resp) or "(No assistant text returned.)"
                if not isinstance(assistant_text, str):
                    assistant_text = str(assistant_text)
                st.session_state.messages.append({"role": "assistant", "content": assistant_text})
                st.session_state.status = "Ready"
                # Clear input box
                st.session_state.main_input = ""
                st.success("Reply received.")
            except Exception as e:
                tb = traceback.format_exc()
                st.session_state.messages.append({"role": "assistant", "content": f"API Error: {e}\n\n{tb}"})
                st.session_state.status = "Error"
                st.error("API Error: " + str(e))

with preview_col:
    st.subheader("Images / Preview")
    if st.session_state.images:
        for idx, ib in enumerate(st.session_state.images):
            try:
                st.image(ib, use_column_width=True)
            except Exception:
                st.write("Could not show preview image.")
    else:
        st.info("Generated images will appear here.")

    # Handle image generation triggers from sidebar buttons
    if 'gen_next' not in st.session_state:
        st.session_state['gen_next'] = None

    # We check the sidebar buttons (gen1/gen4/clear) using their captured booleans
    # (Streamlit executes top-to-bottom so the booleans exist)
    try:
        # read the booleans created in sidebar context
        if 'gen1' in locals():
            pass
    except Exception:
        pass

    # The sidebar variables gen1/gen4/clr_imgs are available — but to be safe, re-evaluate from session state keys used by streamlit
    # We'll simply check the sidebar widget keys directly using st.session_state if present
    if st.session_state.get("image_prompt_input", "").strip():
        prompt_val = st.session_state.get("image_prompt_input", "").strip()
    else:
        prompt_val = ""

    # Determine if any sidebar generation button was pressed (streamlit sets True only on the run where pressed)
    # We rely on the variables gen1, gen4, clr_imgs from the sidebar scope by reading them via st.session_state if created
    gen1_pressed = st.session_state.get("Generate 1 image", False) if False else False  # placeholder to avoid errors

    # Instead, detect button presses by reading the query params or by checking the presence of the ephemeral keys.
    # Simpler approach: re-create minimal image generation UI here so we can handle actions reliably:
    st.markdown("### Quick image generator")
    p_prompt = st.text_input("Quick prompt (preview)", key="quick_prompt_input")
    g1, g4, clr = st.columns([1,1,1])
    with g1:
        if st.button("Generate 1 (preview)"):
            gen_n = 1
            p = st.session_state.get("quick_prompt_input", "").strip()
            if not p:
                st.warning("Please provide a prompt.")
            else:
                st.session_state.status = "Generating image(s)..."
                with st.spinner("Generating..."):
                    try:
                        images_info = []
                        # try modern images API
                        try:
                            if hasattr(client, "images"):
                                resp = client.images.generate(model="gpt-image-1", prompt=p, size="1024x1024", n=gen_n)
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
                                resp = client.Image.create(prompt=p, n=gen_n, size="1024x1024")
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
                                    if fixed:
                                        st.session_state.images.append(fixed)
                                        saved.append(fixed)
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

    with g4:
        if st.button("Generate 4 (preview)"):
            gen_n = 4
            p = st.session_state.get("quick_prompt_input", "").strip()
            if not p:
                st.warning("Please provide a prompt.")
            else:
                st.session_state.status = "Generating image(s)..."
                with st.spinner("Generating..."):
                    try:
                        images_info = []
                        try:
                            if hasattr(client, "images"):
                                resp = client.images.generate(model="gpt-image-1", prompt=p, size="1024x1024", n=gen_n)
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
                                resp = client.Image.create(prompt=p, n=gen_n, size="1024x1024")
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
                                    if fixed:
                                        st.session_state.images.append(fixed)
                                        saved.append(fixed)
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

    with clr:
        if st.button("Clear previews"):
            st.session_state.images = []
            st.success("Cleared preview images.")

    # download images
    if st.session_state.images:
        for i, b in enumerate(st.session_state.images):
            try:
                st.download_button(f"Download image {i+1}", data=b, file_name=f"gen_{i+1}.png", mime="image/png")
            except Exception:
                pass

# -------------------- Footer / utilities --------------------
st.markdown("---")
left, right = st.columns([3,1])
with left:
    st.write("Features: Chat memory, sessions, image generation (1 or 4), PDF & image upload, export sessions.")
with right:
    if st.button("Reset all (chat + saved sessions + previews)"):
        st.session_state.messages = [m for m in st.session_state.messages if m.get("role") == "system"]
        st.session_state.saved_sessions = {}
        st.session_state.images = []
        st.session_state.status = "Ready"
        st.success("Reset complete.")

# End of file
