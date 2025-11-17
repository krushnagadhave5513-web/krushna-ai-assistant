import os
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from werkzeug.utils import secure_filename
import base64
import time
import uuid

API_KEY = os.getenv("API_KEY")  # Set in environment variable
client = OpenAI(api_key=API_KEY)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads/"
os.makedirs("uploads", exist_ok=True)
os.makedirs("chats", exist_ok=True)

# ---------------------------
# Helper – Save Chat
# ---------------------------
def save_chat(user_msg, bot_msg):
    file = "chats/history.txt"
    with open(file, "a", encoding="utf-8") as f:
        f.write(f"USER: {user_msg}\nAI: {bot_msg}\n\n")

# ---------------------------
# TEXT CHAT
# ---------------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data["message"]
    system_prompt = data["system"]

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
    )

    reply = response.choices[0].message.content
    save_chat(message, reply)

    return jsonify({"reply": reply})


# ---------------------------
# IMAGE GENERATION (FIXED)
# ---------------------------
@app.route("/generate-image", methods=["POST"])
def generate_image():
    prompt = request.json["prompt"]

    img = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1024x1024"
    )

    image_base64 = img.data[0].b64_json
    return jsonify({"image": image_base64})


# ---------------------------
# VIDEO GENERATION
# ---------------------------
@app.route("/generate-video", methods=["POST"])
def generate_video():
    prompt = request.json["prompt"]

    result = client.videos.generate(
        model="gpt-video-1",
        prompt=prompt
    )

    video_base64 = result.data[0].b64_video
    return jsonify({"video": video_base64})


# ---------------------------
# FILE ANALYSIS (PDF, IMAGE, VIDEO)
# ---------------------------
@app.route("/analyze-file", methods=["POST"])
def analyze_file():
    file = request.files["file"]
    filename = secure_filename(file.filename)
    filepath = os.path.join("uploads", filename)
    file.save(filepath)

    with open(filepath, "rb") as f:
        file_bytes = f.read()

    file_encoded = base64.b64encode(file_bytes).decode()

    res = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are a file analysis expert."},
            {"role": "user", "content": [
                {"type": "input_file", "data": file_encoded},
                {"type": "text", "text": "Analyze this file."}
            ]}
        ]
    )

    reply = res.choices[0].message.content
    return jsonify({"analysis": reply})


# ---------------------------
# HOME UI
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
