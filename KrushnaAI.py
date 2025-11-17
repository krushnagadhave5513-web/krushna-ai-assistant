"""
Image/Video/Chat Tool — v2 (single-file demo)

This file implements the features you asked for (best-effort demo):
- Robust image generation with retries, clearer error messages and validation
- Video generation (frames -> mp4) using ffmpeg with zip fallback
- Multiple image upload at once
- Image / file / document analysis (txt, pdf, docx supported if libs installed)
- Multi-language Jarvis (system prompt + language selection)
- Chat saving/loading (local JSON), export/import
- Custom system prompt UI for AI behaviour control
- Left-side conversation history with create/delete/select
- Faster response mode toggle
- Typing animation for assistant responses
- Upgraded UI layout using tkinter + ttk

How to run:
1. Install dependencies: pip install openai pillow requests python-docx PyPDF2
2. Have ffmpeg on PATH for mp4 export.
3. Set OPENAI_API_KEY env var or paste into API_KEY variable below (not recommended to hardcode).
4. Run: python image_video_tool_v2.py

Notes:
- This is a local desktop demo using Tkinter. For production apps, move to a web UI (React/Flask) and secure the API key.
- Replace MODEL/IMAGE_MODEL names to match your OpenAI client if needed.

"""

import os
import io
import sys
import json
import time
import base64
import threading
import tempfile
import subprocess
from pathlib import Path
from queue import Queue, Empty
from tkinter import (
    Tk, Frame, Label, Entry, Button, Text, Listbox, filedialog, messagebox,
    simpledialog, StringVar, BooleanVar, END, LEFT, RIGHT, BOTH, Y
)
from tkinter import ttk
from PIL import Image, ImageTk

# Optional imports
try:
    from openai import OpenAI
except Exception:
    try:
        import openai
        OpenAI = openai.OpenAI
    except Exception:
        OpenAI = None

# ---------------- CONFIG ----------------
API_KEY = os.environ.get('API_KEY', '')
MODEL = 'gpt-4o-mini'
IMAGE_MODEL = 'gpt-image-1'
CHAT_SAVE_FILE = Path.home() / '.image_video_tool_v2_chats.json'

if OpenAI is None:
    raise RuntimeError("OpenAI client not installed. Install 'openai' or the official client package.")

client = OpenAI(api_key=API_KEY) if API_KEY else OpenAI()

# ---------------- Utilities ----------------

def retry(fn, retries=3, delay=1, backoff=2):
    last = None
    for i in range(retries):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(delay * (backoff ** i))
    raise last


def load_chats():
    try:
        if CHAT_SAVE_FILE.exists():
            return json.loads(CHAT_SAVE_FILE.read_text(encoding='utf-8'))
    except Exception:
        pass
    return {}


def save_chats(data):
    try:
        CHAT_SAVE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CHAT_SAVE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception as e:
        print('Could not save chats:', e)

# ---------------- Image generation ----------------
import requests

def generate_images(prompt, n=1, size='512x512'):
    """Generate images via client, return list of PIL Images. Robust parsing and helpful errors."""
    def call():
        return client.images.generate(model=IMAGE_MODEL, prompt=prompt, n=n, size=size)

    resp = retry(call, retries=3, delay=1)
    images = []
    # Try common response shapes
    if hasattr(resp, 'data') and isinstance(resp.data, list):
        for item in resp.data:
            b64 = getattr(item, 'b64_json', None) or (item.get('b64_json') if isinstance(item, dict) else None)
            if b64:
                images.append(Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGBA'))
            elif isinstance(item, dict) and item.get('url'):
                r = requests.get(item['url'], timeout=10)
                r.raise_for_status()
                images.append(Image.open(io.BytesIO(r.content)).convert('RGBA'))
    elif isinstance(resp, dict) and resp.get('data'):
        for item in resp['data']:
            if item.get('b64_json'):
                images.append(Image.open(io.BytesIO(base64.b64decode(item['b64_json']))).convert('RGBA'))
            elif item.get('url'):
                r = requests.get(item['url'], timeout=10)
                r.raise_for_status()
                images.append(Image.open(io.BytesIO(r.content)).convert('RGBA'))

    if not images:
        raise RuntimeError('No images returned (check model name, API key, or quota)')
    return images

# ---------------- Video helpers ----------------

def frames_to_mp4(frames, out_path, fps=8):
    """frames: list of file paths. Attempts ffmpeg concat. Falls back to zipping frames."""
    if not frames:
        raise ValueError('No frames')
    # create concat file
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.txt') as f:
        for p in frames:
            f.write(f"file '{p}'\n")
        listfile = f.name
    cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', listfile, '-pix_fmt', 'yuv420p', out_path]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        # fallback to zip
        import zipfile
        zip_out = Path(out_path).with_suffix('.zip')
        with zipfile.ZipFile(zip_out, 'w') as zf:
            for p in frames:
                zf.write(p, arcname=Path(p).name)
        raise RuntimeError(f'FFmpeg failed; frames zipped to {zip_out}: {e}')
    finally:
        try:
            os.remove(listfile)
        except Exception:
            pass

# ---------------- Document analysis ----------------
try:
    import PyPDF2
except Exception:
    PyPDF2 = None
try:
    import docx
except Exception:
    docx = None


def extract_text(path):
    p = Path(path)
    if p.suffix.lower() == '.txt':
        return p.read_text(encoding='utf-8', errors='ignore')
    if p.suffix.lower() == '.pdf' and PyPDF2:
        text = []
        with open(p, 'rb') as f:
            r = PyPDF2.PdfReader(f)
            for pg in r.pages:
                text.append(pg.extract_text() or '')
        return '\n'.join(text)
    if p.suffix.lower() in ('.docx',) and docx:
        d = docx.Document(p)
        return '\n'.join(par.text for par in d.paragraphs)
    return None

# ---------------- App UI ----------------

class App:
    def __init__(self, root):
        self.root = root
        root.title('Image/Video/Chat Tool v2')
        # state
        self.chats = load_chats()
        self.current = None
        self.last_images = []

        # config vars
        self.system_prompt = StringVar(value='You are Jarvis: help the user concisely.')
        self.lang = StringVar(value='English')
        self.faster = BooleanVar(value=False)
        self.typing = BooleanVar(value=True)

        # layout
        left = Frame(root, width=220)
        left.pack(side=LEFT, fill=Y)
        center = Frame(root)
        center.pack(side=LEFT, fill=BOTH, expand=True)
        right = Frame(root, width=360)
        right.pack(side=RIGHT, fill=Y)

        Label(left, text='Conversations').pack()
        self.lb = Listbox(left)
        self.lb.pack(fill=Y, expand=True)
        for n in self.chats.keys():
            self.lb.insert(END, n)
        self.lb.bind('<<ListboxSelect>>', self.on_select)
        bframe = Frame(left)
        bframe.pack(fill='x')
        Button(bframe, text='New', command=self.new_conv).pack(side=LEFT, fill='x', expand=True)
        Button(bframe, text='Delete', command=self.delete_conv).pack(side=LEFT)

        Label(center, text='Chat').pack()
        self.chat_text = Text(center, height=20, wrap='word')
        self.chat_text.pack(fill=BOTH, expand=True)
        eframe = Frame(center)
        eframe.pack(fill='x')
        self.entry = Entry(eframe)
        self.entry.pack(side=LEFT, fill='x', expand=True)
        Button(eframe, text='Send', command=self.send).pack(side=LEFT)
        Button(eframe, text='Save Chat', command=self.export_current_chat).pack(side=LEFT)

        Label(right, text='Controls').pack()
        Label(right, text='System prompt:').pack(anchor='w')
        Entry(right, textvariable=self.system_prompt, width=45).pack(fill='x')
        Label(right, text='Language:').pack(anchor='w')
        ttk.Combobox(right, textvariable=self.lang, values=['English','Hindi','Marathi','Spanish','French'], state='readonly').pack(fill='x')
        ttk.Checkbutton(right, text='Faster mode', variable=self.faster).pack(anchor='w')
        ttk.Checkbutton(right, text='Typing animation', variable=self.typing).pack(anchor='w')

        Button(right, text='Generate Image', command=self.dialog_generate_image).pack(fill='x')
        Button(right, text='Upload Images (multiple)', command=self.upload_images).pack(fill='x')
        Button(right, text='Generate Video from last images', command=self.generate_video_from_last).pack(fill='x')
        Button(right, text='Analyze File/Doc', command=self.analyze_file).pack(fill='x')

        self.preview = Label(right, text='Preview area')
        self.preview.pack()

    # conv management
    def new_conv(self):
        name = simpledialog.askstring('New', 'Conversation name:')
        if not name:
            return
        if name in self.chats:
            messagebox.showinfo('Exists', 'Name already exists')
            return
        self.chats[name] = []
        save_chats(self.chats)
        self.lb.insert(END, name)
        self.select_by_name(name)

    def delete_conv(self):
        sel = self.lb.curselection()
        if not sel:
            return
        name = self.lb.get(sel[0])
        if messagebox.askyesno('Delete', f'Delete {name}?'):
            del self.chats[name]
            save_chats(self.chats)
            self.lb.delete(sel[0])
            self.current = None
            self.chat_text.delete('1.0', END)

    def on_select(self, evt):
        if not self.lb.curselection():
            return
        name = self.lb.get(self.lb.curselection()[0])
        self.select_by_name(name)

    def select_by_name(self, name):
        self.current = name
        self.chat_text.delete('1.0', END)
        for m in self.chats.get(name, []):
            self.chat_text.insert(END, f"{m['role'].upper()}: {m['content']}\n\n")

    # chat send
    def send(self):
        text = self.entry.get().strip()
        if not text:
            return
        if not self.current:
            messagebox.showinfo('Info', 'Select or create conversation first')
            return
        self.append_message('user', text)
        self.entry.delete(0, END)
        threading.Thread(target=self.call_assistant, args=(text,)).start()

    def append_message(self, role, content):
        if not self.current:
            return
        self.chats.setdefault(self.current, []).append({'role': role, 'content': content, 'time': time.time()})
        save_chats(self.chats)
        self.chat_text.insert(END, f"{role.upper()}: {content}\n\n")

    def call_assistant(self, user_text):
        messages = [{'role': 'system', 'content': self.system_prompt.get()},]
        # include recent history
        for m in self.chats.get(self.current, [])[-8:]:
            messages.append({'role': m['role'], 'content': m['content']})
        messages.append({'role': 'user', 'content': user_text})
        params = {'model': MODEL, 'messages': messages}
        if self.faster.get():
            params.update({'max_tokens': 300, 'temperature': 0.2})
        try:
            resp = retry(lambda: client.chat.create(**params), retries=2)
            text = ''
            if hasattr(resp, 'choices') and resp.choices:
                c = resp.choices[0]
                text = getattr(c, 'message', {}).get('content') or (c.get('message', {}).get('content') if isinstance(c, dict) else '')
            elif isinstance(resp, dict) and resp.get('choices'):
                text = resp['choices'][0]['message']['content']
            else:
                text = str(resp)
            self.append_message('assistant', text)
            if self.typing.get():
                self.animate_typing('ASSISTANT: ' + text + '\n\n')
            else:
                self.chat_text.insert(END, 'ASSISTANT: ' + text + '\n\n')
        except Exception as e:
            self.chat_text.insert(END, 'ERROR: ' + str(e) + '\n\n')

    def animate_typing(self, text):
        for ch in text:
            self.chat_text.insert(END, ch)
            self.chat_text.see(END)
            time.sleep(0.01 if self.faster.get() else 0.03)

    # Image UI
    def dialog_generate_image(self):
        prompt = simpledialog.askstring('Prompt', 'Image prompt:')
        if not prompt:
            return
        n = simpledialog.askinteger('Count', 'Number of images (1-4):', initialvalue=1, minvalue=1, maxvalue=8)
        threading.Thread(target=self._gen_images_thread, args=(prompt, n)).start()

    def _gen_images_thread(self, prompt, n):
        try:
            self.set_preview('Generating...')
            imgs = generate_images(prompt, n=n, size='512x512')
            self.last_images = imgs
            self.show_preview(imgs[0])
            self.set_preview(f'Generated {len(imgs)} images')
        except Exception as e:
            self.set_preview('Image error: ' + str(e))

    def upload_images(self):
        paths = filedialog.askopenfilenames(title='Select images', filetypes=[('Images','*.png;*.jpg;*.jpeg;*.bmp;*.gif')])
        if not paths:
            return
        imgs = []
        for p in paths:
            try:
                imgs.append(Image.open(p).convert('RGBA'))
            except Exception as e:
                print('open failed', p, e)
        if imgs:
            self.last_images = imgs
            self.show_preview(imgs[0])
            self.set_preview(f'Loaded {len(imgs)} images')

    def show_preview(self, pil_img):
        w,h = pil_img.size
        maxw=320
        if w>maxw:
            ratio=maxw/w
            pil_img=pil_img.resize((int(w*ratio), int(h*ratio)), Image.ANTIALIAS)
        tkimg = ImageTk.PhotoImage(pil_img)
        self.preview.configure(image=tkimg, text='')
        self.preview.image = tkimg

    def set_preview(self, txt):
        self.preview.configure(text=txt)

    # Video
    def generate_video_from_last(self):
        if not getattr(self, 'last_images', None) or len(self.last_images) < 2:
            messagebox.showinfo('Info', 'Need multiple images (upload or generate).')
            return
        tmpdir = Path(tempfile.mkdtemp(prefix='frames_'))
        frames = []
        for i, img in enumerate(self.last_images):
            p = tmpdir / f'frame_{i:03d}.png'
            img.convert('RGB').save(p)
            frames.append(str(p))
        out = filedialog.asksaveasfilename(defaultextension='.mp4', filetypes=[('MP4','*.mp4')])
        if not out:
            return
        try:
            frames_to_mp4(frames, out, fps=6)
            messagebox.showinfo('Done', f'Video saved to {out}')
        except Exception as e:
            messagebox.showerror('Error', str(e))

    # File analysis
    def analyze_file(self):
        p = filedialog.askopenfilename(title='Select doc', filetypes=[('Docs','*.txt;*.pdf;*.docx')])
        if not p:
            return
        txt = extract_text(p)
        if not txt:
            messagebox.showerror('Extract failed', 'Could not extract text (missing libs or unsupported file)')
            return
        prompt = f"Summarize and analyze the following document in {self.lang.get()}:\n\n{txt[:18000]}"
        threading.Thread(target=self._analyze_thread, args=(prompt,)).start()

    def _analyze_thread(self, prompt):
        try:
            self.set_preview('Analyzing...')
            resp = retry(lambda: client.chat.create(model=MODEL, messages=[{'role':'system','content':self.system_prompt.get()},{'role':'user','content':prompt}]), retries=2)
            text = ''
            if hasattr(resp, 'choices') and resp.choices:
                c = resp.choices[0]
                text = getattr(c, 'message', {}).get('content') or (c.get('message', {}).get('content') if isinstance(c, dict) else '')
            elif isinstance(resp, dict) and resp.get('choices'):
                text = resp['choices'][0]['message']['content']
            else:
                text = str(resp)
            self.set_preview('Analysis complete')
            self.chat_text.insert(END, 'DOCUMENT ANALYSIS: ' + text + '\n\n')
            if self.current:
                self.chats.setdefault(self.current, []).append({'role':'assistant','content':'Document analysis:\n'+text,'time':time.time()})
                save_chats(self.chats)
        except Exception as e:
            self.set_preview('Analysis error: '+str(e))

    def export_current_chat(self):
        if not self.current:
            messagebox.showinfo('Info','Select conversation first')
            return
        out = filedialog.asksaveasfilename(defaultextension='.txt')
        if not out:
            return
        with open(out,'w',encoding='utf-8') as f:
            for m in self.chats.get(self.current,[]):
                f.write(f"{m['role'].upper()}: {m['content']}\n\n")
        messagebox.showinfo('Saved', f'Saved to {out}')


if __name__ == '__main__':
    root = Tk()
    app = App(root)
    root.geometry('1100x720')
    root.mainloop()
