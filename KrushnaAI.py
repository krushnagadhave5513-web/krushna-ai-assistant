"""
Krushna AI Assistant (Flask web app) - GitHub-ready

Features:
- Chat (OpenAI Chat API)
- Image generation (gpt-image-1)
- Video generation (frame-based via images)
- File/document analysis (uploads)
- Chat saving (local JSON)
- Custom system prompt and language selector
- Simple front-end (templates + static)

Usage (local):
1. pip install -r requirements.txt
2. set environment variable API_KEY (Windows):
   set API_KEY=sk-...
   (PowerShell) $env:API_KEY="sk-..."
   (Linux/mac) export API_KEY="sk-..."
3. python app.py
4. Open http://127.0.0.1:5000

NOTE: For production, deploy to Render/Heroku and set env var API_KEY there.
"""

import os
import io
import json
import base64
import tempfile
import traceback
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

# OpenAI client (supports both new and classic packages)
API_KEY = os.getenv("API_KEY")  # <<-- set this in your environment
if not API_KEY:
    raise RuntimeError("Set environment variable API_KEY before running the app.")

# Try new official client first
try:
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY)
except Exception:
    import openai
    openai.api_key = API_KEY
    client = openai

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
Path(app.config["UPLOAD_FOLDER"]).mkdir(exist_ok=True)
Path("chats").mkdir(exist_ok=True)

CHAT_STORE = Path("chats/chats.json")
if not CHAT_STORE.exists():
    CHAT_STORE.write_text(json.dumps({}), encoding="utf-8")

# ---------------- helpers ----------------
def load_chats():
    try:
        return json.loads(CHAT_STORE.read_text(encoding="utf-8") or "{}")
    except Exception:
        return {}

def save_chats(obj):
    CHAT_STORE.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def add_chat(conv_name, role, content):
    data = load_chats()
    data.setdefault(conv_name, []).append({"role": role, "content": content})
    save_chats(data)

# ---------------- routes ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chats", methods=["GET"])
def api_get_chats():
    return jsonify(load_chats())

@app.route("/api/chat", methods=["POST"])
def api_chat():
    payload = request.json or {}
    conv = payload.get("conversation", "default")
    system_prompt = payload.get("system_prompt", "You are a helpful assistant.")
    user_msg = payload.get("message", "")
    lang = payload.get("language", "English")

    # build messages
    messages = [{"role": "system", "content": f"{system_prompt} Respond in {lang}."}]
    # include conversation history (last N messages)
    chats = load_chats()
    for m in chats.get(conv, [])[-12:]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_msg})

    try:
        # Try new client chat API
        if hasattr(client, "chat") and hasattr(client.chat, "create"):
            resp = client.chat.create(model="gpt-4o-mini", messages=messages)
            # parse response
            text = ""
            if hasattr(resp, "choices") and resp.choices:
                c = resp.choices[0]
                text = getattr(c, "message", {}).get("content") or (c.get("message", {}).get("content") if isinstance(c, dict) else "")
            else:
                text = str(resp)
        else:
            # fallback to classic openai.ChatCompletion.create
            resp = client.ChatCompletion.create(model="gpt-4o-mini", messages=messages)
            text = resp["choices"][0]["message"]["content"]
        add_chat(conv, "user", user_msg)
        add_chat(conv, "assistant", text)
        return jsonify({"reply": text})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/generate-image", methods=["POST"])
def api_generate_image():
    """Generates image(s) using gpt-image-1. Returns base64 PNG(s)."""
    data = request.json or {}
    prompt = data.get("prompt", "")
    count = int(data.get("n", 1))
    size = data.get("size", "1024x1024")
    try:
        # New client style
        if hasattr(client, "images") and hasattr(client.images, "generate"):
            resp = client.images.generate(model="gpt-image-1", prompt=prompt, n=count, size=size)
            images_b64 = []
            for item in resp.data:
                b64 = getattr(item, "b64_json", None) or (item.get("b64_json") if isinstance(item, dict) else None)
                if b64:
                    images_b64.append(b64)
                elif isinstance(item, dict) and item.get("url"):
                    # fetch the url and convert to base64
                    import requests
                    r = requests.get(item["url"])
                    r.raise_for_status()
                    images_b64.append(base64.b64encode(r.content).decode())
        else:
            # fallback: classic openai.Image.create
            resp = client.Image.create(prompt=prompt, n=count, size=size)
            images_b64 = [d["b64_json"] for d in resp["data"]]
        return jsonify({"images": images_b64})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/generate-video", methods=["POST"])
def api_generate_video():
    """
    Generate video by creating multiple frames from image generator and returning a downloadable MP4.
    Requires ffmpeg available on server for the concat method. If not available, returns zipped frames.
    """
    data = request.json or {}
    prompt = data.get("prompt", "")
    frames_count = int(data.get("frames", 8))
    size = data.get("size", "512x512")

    tmpdir = Path(tempfile.mkdtemp(prefix="frames_"))
    frames_paths = []
    try:
        for i in range(frames_count):
            frame_prompt = f"{prompt} (frame {i+1} of {frames_count})"
            # generate one image
            if hasattr(client, "images") and hasattr(client.images, "generate"):
                resp = client.images.generate(model="gpt-image-1", prompt=frame_prompt, n=1, size=size)
                item = resp.data[0]
                b64 = getattr(item, "b64_json", None) or (item.get("b64_json") if isinstance(item, dict) else None)
                if b64:
                    data_bytes = base64.b64decode(b64)
                else:
                    # fetch url
                    import requests
                    url = item.get("url")
                    r = requests.get(url); r.raise_for_status()
                    data_bytes = r.content
            else:
                resp = client.Image.create(prompt=frame_prompt, n=1, size=size)
                data_bytes = base64.b64decode(resp["data"][0]["b64_json"])
            p = tmpdir / f"frame_{i:03d}.png"
            p.write_bytes(data_bytes)
            frames_paths.append(str(p))

        # try ffmpeg concat
        out_mp4 = tmpdir / "output.mp4"
        try:
            listfile = tmpdir / "list.txt"
            with open(listfile, "w", encoding="utf-8") as f:
                for p in frames_paths:
                    f.write(f"file '{p}'\n")
            cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listfile), "-pix_fmt", "yuv420p", str(out_mp4)]
            import subprocess
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return send_file(str(out_mp4), as_attachment=True, download_name="generated_video.mp4")
        except Exception:
            # fallback: zip frames and return base64 zip
            import zipfile, io
            zipbuf = io.BytesIO()
            with zipfile.ZipFile(zipbuf, "w") as z:
                for p in frames_paths:
                    z.write(p, arcname=Path(p).name)
            zip_b64 = base64.b64encode(zipbuf.getvalue()).decode()
            return jsonify({"frames_zip_b64": zip_b64})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        # cleanup left to server (optional)
        pass

@app.route("/api/analyze-file", methods=["POST"])
def api_analyze_file():
    """
    Upload a file and analyze (text extraction + send to chat model for summary).
    Supports txt, pdf (PyPDF2), docx (python-docx).
    """
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    f = request.files["file"]
    filename = secure_filename(f.filename)
    dest = Path(app.config["UPLOAD_FOLDER"]) / filename
    f.save(dest)

    # extract text
    suffix = dest.suffix.lower()
    text = None
    if suffix == ".txt":
        text = dest.read_text(encoding="utf-8", errors="ignore")
    elif suffix == ".pdf":
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(str(dest))
            pages = [p.extract_text() or "" for p in reader.pages]
            text = "\n".join(pages)
        except Exception:
            text = None
    elif suffix == ".docx":
        try:
            import docx
            d = docx.Document(str(dest))
            text = "\n".join(p.text for p in d.paragraphs)
        except Exception:
            text = None

    if not text:
        return jsonify({"error": "Could not extract text from file (missing libs or unsupported)"}), 400

    # send to chat model for analysis
    try:
        system = "You are a document analysis assistant."
        messages = [{"role": "system", "content": system}, {"role": "user", "content": f"Please summarize and analyze the following document:\n\n{text[:20000]}"}]
        if hasattr(client, "chat") and hasattr(client.chat, "create"):
            resp = client.chat.create(model="gpt-4o-mini", messages=messages)
            content = resp.choices[0].message.content if hasattr(resp.choices[0], "message") else resp.choices[0]["message"]["content"]
        else:
            resp = client.ChatCompletion.create(model="gpt-4o-mini", messages=messages)
            content = resp["choices"][0]["message"]["content"]
        return jsonify({"analysis": content})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ---------------- run ----------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
