import streamlit as st
from openai import OpenAI
from PIL import Image
import io

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Krushna AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------------------------------------
# API KEY (your variable name)
# ---------------------------------------------------------
API_KEY = st.secrets["API_KEY"]   # Streamlit Cloud secret
client = OpenAI(api_key=API_KEY)

# ---------------------------------------------------------
# INITIAL STATES
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = []

# ---------------------------------------------------------
# AI CHAT FUNCTION
# ---------------------------------------------------------
def ai_chat(history):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=history
    )
    return response.choices[0].message.content

# ---------------------------------------------------------
# SIDEBAR (chat history + memory)
# ---------------------------------------------------------
with st.sidebar:
    st.title("📚 Chat History")

    if st.button("🆕 New Chat"):
        st.session_state.messages = []
        st.session_state.memory = []
        st.rerun()

    st.write("### Stored Memory:")
    if len(st.session_state.memory) == 0:
        st.info("No memory stored")
    else:
        for m in st.session_state.memory:
            st.write(f"• {m}")

# ---------------------------------------------------------
# MAIN TITLE
# ---------------------------------------------------------
st.title("🤖 Krushna AI Assistant")
st.write("Ask anything below!")

# ---------------------------------------------------------
# SHOW CHAT HISTORY
# ---------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------------------------------------------------------
# 📩 USER INPUT (ENTER = SEND)
# ---------------------------------------------------------
user_input = st.chat_input("Type your message... (Press Enter to send)")

if user_input is not None and user_input.strip() != "":
    # save to memory
    st.session_state.memory.append(user_input)

    # save user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # AI response
    with st.chat_message("assistant"):
        reply = ai_chat(st.session_state.messages)
        st.write(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

# ---------------------------------------------------------
# FILE UPLOAD (PDF / IMAGE)
# ---------------------------------------------------------
uploaded = st.file_uploader("Upload image or PDF", type=["png", "jpg", "jpeg", "pdf"])

if uploaded:
    st.write("📄 File received!")

    if uploaded.type.startswith("image"):
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Image")

        # Convert image to bytes for AI
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes = img_bytes.getvalue()

        with st.chat_message("assistant"):
            st.write("Analyzing image...")

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": "Describe the image"},
                    {"role": "user", "content": [{"type": "input_image", "image": img_bytes}]}
                ]
            )
            st.write(response.choices[0].message.content)

    else:
        st.warning("PDF support added but basic — text extraction version coming soon!")

