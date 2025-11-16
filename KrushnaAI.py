# KrushnaAI.py
# Single-file Streamlit app — Krushna AI Assistant
# Features: Chat + Memory + Sessions + Image generation (1 or multiple)
# PDF/Image upload (text extraction), background alpha handling, export sessions.
# Requires: streamlit, openai, pillow, requests, PyPDF2
# Use Streamlit Secret: API_KEY = "sk-..."

import streamlit as st
from openai import OpenAI
from PIL import Image, ImageOps
import io, os, time, json, base64, tempfile, traceback
import requests

# Optional PDF lib
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

# -------------------------
# Helpers
# -------------------------
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
                continue
            if not isinstance(m, dict):
                out.append({"role": "user", "content": str(m)})
                continue
            role = m.get("role", "user") or "user"
            content = m.get("content", "") or ""
            out.append({"role": role, "content": content})
        except Exception:
            out.append({"role": "user", "content": str(m)})
    return out

def extract_assistant_text_from_raw(raw):
    try:
        # object-like
        if hasattr(raw, "choices") and len(raw.choices) > 0:
            c0 = raw.choices[0]
            # prefer .message.content (new SDK)
            try:
                if hasattr(c0, "message") and c0.message:
                    msg = c0.message
                    if hasattr(msg, "content"):
                        return msg.content or ""
                    # dict-like
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
            # fallback to text attr
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
        # final fallback
        return str(raw)
    except Exception:
        try:
            return str(raw)
        except Exception:
            return ""

def save_session_local(session_name, messages):
    # Save JSON in /tmp or in-session download ready
    fname = f"{session_name}_{int(time.time())}.json"
    tmp = tempfile.gettempdir()
    path = os.path.join(tmp, fname)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
        return path
    except Exception:
        return None

def pil_fix_alpha_make_white(im: Image.Image):
    # If image has alpha, place it over white background to avoid black alpha issues
    try:
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            bg.paste(im, mask=im.split()[-1])  # paste using alpha channel
            return bg.convert("RGB")
        else:
            return im.convert("RGB")
    except Exception:
        return im

# -------------------------
# Streamlit UI setup
# -------------------------
st.set_page_config(page_title="Krushna AI Assistant", page_icon="🤖", layout="wide")
st.title("🤖 Krushna AI Assistant")
st.write("Chat, generate images, upload PDFs/images, and keep sessions — powered by OpenAI.")

# -------------------------
# Load API key (Streamlit Secret name: API_KEY)
# -------------------------
if "API_KEY" not in st.secrets:
    st.error("Missing secret: API_KEY. Set it in Streamlit app settings before using the assistant.")
    st.stop()

API_KEY = st.secrets["API_KEY"]

# Create client once in session state
if "client" not in st.session_state:
    try:
        st.session_state.client = OpenAI(api_key=API_KEY)
    except Exception as e:
        st.error("Could not initialize OpenAI client: " + str(e))
        st.stop()

client = st.session_state.client

# -------------------------
# Session state initialization
# -------------------------
if "messages" not in st.session_state:
    # base system role to keep assistant consistent
    st.session_state.messages = [{"role": "system", "content": "You are Jarvis 2.0, helpful assistant for Krushna. Keep answers concise and code-friendly."}]
if "saved_sessions" not in st.session_state:
    st.session_state.saved_sessions = {}  # name -> messages
if "images" not in st.session_state:
    st.session_state.images = []  # paths or in-memory bytes
if "status" not in st.session_state:
    st.session_state.status = "Ready"

# Sidebar: sessions, controls, uploads
with st.sidebar:
    st.header("Sessions")
    # list saved session names
    if st.session_state.saved_sessions:
        sel = st.selectbox("Load saved session", options=["-- choose --"] + list(st.session_state.saved_sessions.keys()))
        if sel and sel != "-- choose --":
            if st.button("Load Session"):
                st.session_state.messages = sanitize_messages(st.session_state.saved_sessions[sel])
                st.success(f"Loaded session: {sel}")
                st.experimental_rerun()
        if st.button("Clear saved sessions"):
            st.session_state.saved_sessions = {}
            st.success("Cleared saved sessions")
    else:
        st.info("No saved sessions yet. Save current conversation below.")

    st.markdown("---")
    st.write("Save / Export")
    sess_name = st.text_input("Session name", value=f"session_{time.strftime('%Y%m%d_%H%M%S')}")
    if st.button("Save current session"):
        try:
            st.session_state.saved_sessions[sess_name] = list(st.session_state.messages)
            st.success(f"Saved: {sess_name}")
        except Exception as e:
            st.error("Save failed: " + str(e))
    if st.button("Export current session (.txt)"):
        # prepare text
        lines = []
        for m in st.session_state.messages:
            role = m.get("role", "user")
            prefix = "You: " if role == "user" else ("Jarvis: " if role == "assistant" else f"{role.capitalize()}: ")
            lines.append(prefix + m.get("content", ""))
        txt = "\n\n".join(lines)
        st.download_button("Download transcript", txt, file_name=f"transcript_{int(time.time())}.txt", mime="text/plain")

    st.markdown("---")
    st.header("Upload")
    uploaded_image = st.file_uploader("Upload image (PNG/JPG)", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=False)
    st.markdown("---")
    st.write("Image generation")
    image_prompt = st.text_input("Image prompt (sidebar)")
    cols = st.columns([1,1,1])
    with cols[0]:
        if st.button("Generate image (1)"):
            st.session_state._gen_image_request = {"prompt": image_prompt, "n":1}
    with cols[1]:
        if st.button("Generate 4 images"):
            st.session_state._gen_image_request = {"prompt": image_prompt, "n":4}
    with cols[2]:
        if st.button("Clear preview images"):
            st.session_state.images = []
    st.markdown("---")
    st.write("Status")
    st.info(st.session_state.status)

# Main layout: two columns (chat + image preview)
chat_col, preview_col = st.columns([2.2, 1])

with chat_col:
    # chat display
    st.subheader("Conversation")
    # show history in chat-like blocks
    for m in st.session_state.messages:
        role = m.get("role","user")
        content = m.get("content","")
        # use chat_message if available (Streamlit >=1.23)
        try:
            with st.chat_message(role):
                st.markdown(content)
        except Exception:
            # fallback simple display
            if role == "user":
                st.markdown(f"**You:** {content}")
            elif role == "assistant":
                st.markdown(f"**Jarvis:** {content}")
            else:
                st.markdown(f"**{role}:** {content}")

    # input area
    input_col, send_col = st.columns([7,1])
    with input_col:
        user_input = st.text_input("Type your message ...", key="chat_input")
    with send_col:
        send_clicked = st.button("Send")

    # quick actions row
    q1, q2, q3 = st.columns([1,1,1])
    with q1:
        if st.button("Send last"):
            # resend last user message
            last = None
            for m in reversed(st.session_state.messages):
                if m.get("role") == "user":
                    last = m.get("content")
                    break
            if last:
                user_input = last
                st.session_state.chat_input = last
                send_clicked = True
    with q2:
        if st.button("Clear chat"):
            st.session_state.messages = [m for m in st.session_state.messages if m.get("role")=="system"]
            st.experimental_rerun()
    with q3:
        if st.button("Save snapshot (.json)"):
            p = save_session_local("snapshot", st.session_state.messages)
            if p:
                with open(p,"rb") as f:
                    st.download_button("Download JSON snapshot", f, file_name=os.path.basename(p))
            else:
                st.error("Could not save snapshot.")

    # handle uploaded image
    if uploaded_image is not None:
        try:
            img = Image.open(uploaded_image)
            img_fixed = pil_fix_alpha_make_white(img)
            buffered = io.BytesIO()
            img_fixed.save(buffered, format="PNG")
            b64 = base64.b64encode(buffered.getvalue()).decode()
            st.session_state.messages.append({"role":"user","content":f"[Uploaded Image] {uploaded_image.name}"})
            st.success(f"Uploaded image: {uploaded_image.name}")
        except Exception as e:
            st.error("Image upload error: " + str(e))

    # handle uploaded pdf
    if uploaded_pdf is not None:
        st.session_state.messages.append({"role":"user","content":f"[Uploaded PDF] {uploaded_pdf.name}"})
        if PyPDF2 is None:
            st.warning("PyPDF2 not installed on server — PDF text extraction unavailable.")
        else:
            try:
                reader = PyPDF2.PdfReader(uploaded_pdf)
                text = ""
                for p in reader.pages[:3]:
                    try:
                        text += p.extract_text() or ""
                    except Exception:
                        pass
                snippet = text[:1000] + ("..." if len(text)>1000 else "")
                if snippet.strip():
                    st.session_state.messages.append({"role":"assistant","content":f"[PDF extract]\n{snippet}"})
                else:
                    st.session_state.messages.append({"role":"assistant","content":"[PDF] Could not extract text."})
            except Exception as e:
                st.session_state.messages.append({"role":"assistant","content":"[PDF] Error: " + str(e)})

    # process send (synchronous UI, but we mark status)
    if send_clicked and user_input and user_input.strip():
        st.session_state.status = "Contacting Jarvis..."
        st.experimental_rerun()  # rerun to show status first

    # On rerun: check if there is unsent input in session_state.chat_input and handle
    if "chat_input" in st.session_state and st.session_state.chat_input and send_clicked:
        pass  # handled above; kept for compatibility

    # We'll create a small background job style handler via a button click pattern: use state flag
    if "pending_user" not in st.session_state:
        st.session_state.pending_user = None

    # If user pressed Send (we'll use st.session_state.chat_input)
    if send_clicked:
        text = st.session_state.get("chat_input", "") or user_input
        if text and text.strip():
            st.session_state.messages.append({"role":"user","content":text})
            # call OpenAI
            st.session_state.status = "Jarvis is thinking..."
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=st.session_state.messages,
                    max_tokens=800,
                    temperature=0.4
                )
                assistant_text = extract_assistant_text_from_raw(resp) or "(No assistant text returned.)"
                # normalize: ensure string
                if not isinstance(assistant_text, str):
                    assistant_text = str(assistant_text)
                st.session_state.messages.append({"role":"assistant","content":assistant_text})
                st.session_state.status = "Ready"
                # clear input
                st.session_state.chat_input = ""
                st.experimental_rerun()
            except Exception as e:
                tb = traceback.format_exc()
                st.session_state.messages.append({"role":"assistant","content":f"API Error: {e}\n\n{tb}"})
                st.session_state.status = "Error"
                st.experimental_rerun()

with preview_col:
    st.subheader("Images / Preview")
    # show in-session images
    if st.session_state.images:
        for i, imdata in enumerate(st.session_state.images):
            try:
                if isinstance(imdata, bytes):
                    st.image(imdata, use_column_width=True)
                else:
                    st.image(imdata, use_column_width=True)
            except Exception:
                st.write("Could not show image preview.")
    else:
        st.info("Generated images will appear here.")

    # Image generation handler triggered from sidebar
    if "_gen_image_request" in st.session_state and st.session_state._gen_image_request:
        req = st.session_state.pop("_gen_image_request")
        p = req.get("prompt","")
        n = int(req.get("n",1))
        if not p or not p.strip():
            st.warning("Enter an image prompt in the sidebar first.")
        else:
            st.session_state.status = "Generating image(s)..."
            try:
                # Try modern images API first
                images_info = []
                try:
                    if hasattr(client, "images"):
                        resp = client.images.generate(model="gpt-image-1", prompt=p, size="1024x1024", n=n)
                        # resp.data -> list of objects with b64_json or url
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

                # fallback to legacy
                if not images_info:
                    try:
                        resp = client.Image.create(prompt=p, n=n, size="1024x1024")
                        for d in resp.get("data", []):
                            images_info.append({"b64": d.get("b64_json"), "url": d.get("url")})
                    except Exception:
                        pass

                saved = []
                for i, info in enumerate(images_info):
                    img_bytes = None
                    if info.get("url"):
                        try:
                            r = requests.get(info["url"], timeout=15)
                            r.raise_for_status()
                            img_bytes = r.content
                        except Exception:
                            # fallback to b64
                            b64 = info.get("b64")
                            if b64:
                                img_bytes = base64.b64decode(b64)
                    elif info.get("b64"):
                        img_bytes = base64.b64decode(info.get("b64"))
                    if img_bytes:
                        # fix alpha if needed
                        try:
                            im = Image.open(io.BytesIO(img_bytes))
                            im_fixed = pil_fix_alpha_make_white(im)
                            buff = io.BytesIO()
                            im_fixed.save(buff, format="PNG")
                            img_bytes = buff.getvalue()
                        except Exception:
                            pass
                        st.session_state.images.append(img_bytes)
                        saved.append(img_bytes)
                if saved:
                    st.success(f"Saved {len(saved)} image(s).")
                else:
                    st.warning("No images returned by API.")
            except Exception as e:
                st.error("Image generation error: " + str(e))
            finally:
                st.session_state.status = "Ready"
                st.experimental_rerun()

    # allow download of previewed images
    if st.session_state.images:
        for idx, img_bytes in enumerate(st.session_state.images):
            try:
                st.download_button(f"Download image {idx+1}", data=img_bytes, file_name=f"gen_{idx+1}.png", mime="image/png")
            except Exception:
                pass

# Footer / small help area
st.markdown("---")
col1, col2 = st.columns([3,1])
with col1:
    st.write("Features: ChatGPT-style conversation, memory (saved sessions), image generation (1 or 4), PDF & image upload, export sessions.")
with col2:
    if st.button("Reset everything (clear sessions & chat)"):
        st.session_state.messages = [m for m in st.session_state.messages if m.get("role")=="system"]
        st.session_state.saved_sessions = {}
        st.session_state.images = []
        st.success("Reset complete. System message retained.")
        st.experimental_rerun()
