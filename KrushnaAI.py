import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Krushna AI Assistant", page_icon="🤖")

API_KEY = st.secrets["API_KEY"]
client = OpenAI(api_key=API_KEY)

st.title("🤖 Krushna AI Assistant")
st.write("Built using **Streamlit + OpenAI API**")

prompt = st.text_input("Type your message:")

if st.button("Send"):
    if prompt.strip() == "":
        st.warning("Please enter a message!")
    else:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        reply = response.choices[0].message["content"]
        st.success(reply)
