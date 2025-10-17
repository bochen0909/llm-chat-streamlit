import argparse
import os
import time
import streamlit as st
from ai_chat_streamlit.chat.session import Session as ChatSession


def parse_args(args):
    parser = argparse.ArgumentParser('Chatbot Streamlit App')
    parser.add_argument('-c', '--config', help='config file',
                        required=False, default='config.yaml')
    return parser.parse_args(args)


args = parse_args(None)
chat_config_file = args.config


st.set_page_config(layout="centered")

if 'chat_session' not in st.session_state:
    st.session_state.chat_session = ChatSession(
        config_file=chat_config_file, on_state_changed=st.rerun)

chat_session = st.session_state.chat_session

st.title(f"{chat_session.current_model} Chat")


def model_selector():
    return st.selectbox('Model:', chat_session.models, index=chat_session.models.index(chat_session.current_model))


def file_selector():
    history_list = chat_session.history_files
    history_list += [('New', None, None)]
    files = [t[0] for t in history_list]
    try:
        idx = files.index(chat_session.current_history)
    except ValueError:
        idx = files.index('New')

    def _format(t):
        fn, mtime, first_line = t
        if fn == 'New':
            return fn
        mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
        return f"[{mtime}] {first_line[:50]}"
    selected = st.selectbox('Session:', history_list,
                            index=idx, format_func=_format)
    return selected[0]


if True:
    chat_session.set_model(model_selector())
if True:
    filename = file_selector()
    chat_session.set_history_file(None if filename == 'New' else filename)

if not chat_session.is_new_session:
    st.write(
        f"Using history file: {os.path.join(chat_session.history_folder, chat_session.current_history)}")

client = chat_session.chat_model(chat_session.current_model)


for message in chat_session.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    chat_session.add_message("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.chat_message("assistant"):
            stream = client.stream(chat_session.messages)
            response = st.write_stream(stream)
    except Exception as e:
        print(f"Error: {e}")
        client.reset()
        with st.chat_message("assistant"):
            stream = client.stream(chat_session.messages)
            response = st.write_stream(stream)
            
    chat_session.add_message("assistant", response)
    chat_session.save_history()
