# KRUSHNA AI ASSISTANT — FULL CODE (IMAGE + VIDEO + ANALYSIS + CHAT SAVE + MULTIUPLOAD)
# ---------------------------------------------------------------
# IMPORTANT:
# - Replace YOUR_API_KEY with your OpenAI API Key
# - Install dependencies:
#   pip install openai pillow imageio python-docx PyPDF2 customtkinter
# - Install ffmpeg for video export
# ---------------------------------------------------------------

import base64
import os
import json
import imageio
import customtkinter as ctk
from tkinter.filedialog import askopenfilename, askopenfilenames
from openai import OpenAI
from PIL import Image, ImageTk

# API CLIENT
API_KEY = "YOUR_API_KEY"
client = OpenAI(api_key=API_KEY)

# SYSTEM PROMPT
SYSTEM_PROMPT = "You are Krushna AI Assistant: fast, smart, helpful, respectful, with short modern responses."

chat_history = []

# ---------------------------------------------------------------
# IMAGE GENERATION
# ---------------------------------------------------------------
def generate_image(prompt):
    try:
        result = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024",
            n=1
        )
        img_data = base64.b64decode(result.data[0].b64_json)

        with open("generated_image.png", "wb") as f:
            f.write(img_data)
        return "Image saved as generated_image.png"
    except Exception as e:
        return f"Image generation failed: {e}"

# ---------------------------------------------------------------
# VIDEO GENERATION
# ---------------------------------------------------------------
def generate_video(prompt):
    try:
        result = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024",
            n=8
        )
        frames = []

        for i, img in enumerate(result.data):
            img_bytes = base64.b64decode(img.b64_json)
            filename = f"frame_{i}.png"
            with open(filename, "wb") as f:
                f.write(img_bytes)
            frames.append(imageio.imread(filename))

        imageio.mimsave("output.mp4", frames, fps=4)
        return "Video saved as output.mp4"
    except Exception as e:
        return f"Video generation failed: {e}"

# ---------------------------------------------------------------
# FILE ANALYSIS
# ---------------------------------------------------------------
def analyze_file(path):
    try:
        with open(path, "rb") as f:
            result = client.responses.create(
                model="gpt-4o-mini",
                input="Analyze this file in detail",
                files={"file": f}
            )
        return result.output_text
    except Exception as e:
        return f"File analysis failed: {e}"

# ---------------------------------------------------------------
# MULTI-LANGUAGE JARVIS
# ---------------------------------------------------------------
def ask_jarvis(prompt, lang="en"):
    result = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": f"Respond in {lang}."},
            {"role": "user", "content": prompt}
        ]
    )
    return result.choices[0].message["content"]

# ---------------------------------------------------------------
# CHAT SAVE/LOAD
# ---------------------------------------------------------------
def save_chat():
    with open("chat_history.json", "w") as f:
        json.dump(chat_history, f, indent=4)


def load_chat():
    try:
        with open("chat_history.json", "r") as f:
            return json.load(f)
    except:
        return []

# ---------------------------------------------------------------
# MAIN AI RESPONSE
# ---------------------------------------------------------------
def ask_ai(msg):
    chat_history.append({"user": msg})

    res = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": msg}
        ]
    )

    reply = res.choices[0].message["content"]
    chat_history.append({"assistant": reply})
    return reply

# ---------------------------------------------------------------
# TYPING ANIMATION
# ---------------------------------------------------------------
def type_text(widget, text):
    widget.configure(state="normal")
    widget.delete("0.0", "end")
    for char in text:
        widget.insert("end", char)
        widget.update()
        widget.after(10)
    widget.configure(state="disabled")

# ---------------------------------------------------------------
# UI (CustomTkinter Modern UI)
# ---------------------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("KRUSHNA AI ASSISTANT")
app.geometry("1000x700")

# LEFT SIDEBAR — CHAT HISTORY
sidebar = ctk.CTkFrame(app, width=220, corner_radius=12)
sidebar.pack(side="left", fill="y", padx=10, pady=10)

history_box = ctk.CTkTextbox(sidebar, width=200, height=650)
history_box.pack(padx=10, pady=10)

# MAIN CHAT AREA
main_frame = ctk.CTkFrame(app, corner_radius=12)
main_frame.pack(side="right", expand=True, fill="both", padx=10, pady=10)

chat_box = ctk.CTkTextbox(main_frame, width=700, height=520)
chat_box.pack(pady=10)

entry = ctk.CTkEntry(main_frame, width=500, height=45, placeholder_text="Type your message...")
entry.pack(side="left", padx=10)

# ---------------------------------------------------------------
# BUTTON FUNCTIONS
# ---------------------------------------------------------------
def send_message():
    msg = entry.get()
    entry.delete(0, "end")
    if not msg:
        return

    history_box.insert("end", f"You: {msg}\n")

    reply = ask_ai(msg)
    type_text(chat_box, reply)

    history_box.insert("end", f"AI: {reply}\n\n")


send_btn = ctk.CTkButton(main_frame, text="Send", command=send_message)
send_btn.pack(side="left")

# ---------------------------------------------------------------
# EXTRA BUTTONS — IMAGE, VIDEO, ANALYSIS
# ---------------------------------------------------------------
def do_image():
    prompt = entry.get()
    entry.delete(0, "end")
    result = generate_image(prompt)
    history_box.insert("end", f"[Image] {result}\n")


def do_video():
    prompt = entry.get()
    entry.delete(0, "end")
    result = generate_video(prompt)
    history_box.insert("end", f"[Video] {result}\n")


def do_file_analysis():
    file = askopenfilename()
    if not file:
        return
    result = analyze_file(file)
    history_box.insert("end", f"[Analysis]\n{result}\n")


def do_multiupload():
    files = askopenfilenames()
    history_box.insert("end", f"Uploaded {len(files)} files.\n")

image_btn = ctk.CTkButton(sidebar, text="Generate Image", command=do_image)
image_btn.pack(pady=5)

video_btn = ctk.CTkButton(sidebar, text="Generate Video", command=do_video)
video_btn.pack(pady=5)

file_btn = ctk.CTkButton(sidebar, text="Analyze File", command=do_file_analysis)
file_btn.pack(pady=5)

multi_btn = ctk.CTkButton(sidebar, text="Multi Upload", command=do_multiupload)
multi_btn.pack(pady=5)

# ---------------------------------------------------------------
# RUN APP
# ---------------------------------------------------------------
app.mainloop()
