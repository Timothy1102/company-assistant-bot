import streamlit as st
import bot

with st.sidebar:
    "[View source code](https://github.com/Timothy1102/company-assistant-bot)"

st.title("💬 Kobizo Assistant 🤖")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I assist you today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    kobizoRes = bot.generate_response(prompt)
    st.session_state.messages.append({"role": "bot", "content": kobizoRes})
    st.chat_message("assistant").write(kobizoRes)
