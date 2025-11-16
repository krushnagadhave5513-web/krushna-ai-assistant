import streamlit as st
from openai import OpenAI

# ------------------------------
# LOAD API KEY FROM STREAMLIT SECRETS
# ------------------------------
API_KEY = st.secrets["API_KEY"]
client = OpenAI(api_key=API_KEY)

# ------------------------------
# STREAMLIT UI
# ------------------------------
st.set_page_config(page_title="Krushna AI Assistant")
st.title("🤖 Krushna AI Assistant")
st.write("Built using Streamlit + OpenAI API")

# ------------------------------
# USER INPUT
# ------------------------------
user_input = st.text_input("Type your message:")

# ------------------------------
# SEND MESSAGE
# ------------------------------
if st.button("Send"):
    if not user_input.strip():
        st.warning("Please type something before sending.")
    else:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": user_input}
                ]
            )

            # CORRECT FORMAT
            reply = response.choices[0].message.content

            st.success(reply)

        except Exception as e:
            st.error("Error: " + str(e))
