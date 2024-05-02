import streamlit as st
import random
import time
from llm_model import app  # 'App' is the workflow compiled in chatbot_app2.py

def add_vertical_space(num_lines):
    for _ in range(num_lines):
        st.write("")

def colored_header(label, description, color_name):
    st.markdown(f"""
        <div style="background-color:{color_name};padding:10px;border-radius:5px">
            <h2 style="color:white;text-align:center;">{label}</h2>
            <p style="color:white;text-align:center;">{description}</p>
        </div>
    """, unsafe_allow_html=True)

@st.cache_data
def get_initial_session_state():
    return {
        "messages": [],
        "cached_responses": {},
        "context": ""
    }

def run_chat(question, context):
    inputs = {"question": question, "context": context}
    response_text = ""
    try:
        for output in app.stream(inputs):
            for key, value in output.items():
                if key == "generate":
                    response_text = value["generation"]
                    break
            if response_text:
                break
    except Exception as e:
        response_text = str(e)
    return response_text

def simulate_typing(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

def main():
    st.set_page_config(page_icon=None, page_title="Climate Change Chatbot", layout="centered", initial_sidebar_state="auto", menu_items=None)

    with st.sidebar:
        st.title(' 🌍 Climate Change App')
        st.markdown('''
            ## About
            This app is an AI-powered chatbot built using:
            - Streamlit
            - Command R- Plus AI model
            - The purpose of this app is to educate individuals about climate change and foster a community of informed citizens. It provides accurate information and resources about climate change and its impacts, and encourages users to take action in their own communities.
        ''')
        add_vertical_space(5)
        st.write('Made By 🌳 Climate Change Communities')

    col1, col2 = st.columns([1, 8])

    with col1:
        add_vertical_space(1)
        st.image("C:\\Users\\luist\\Downloads\\CCCicon.png", width=100)

    with col2:
        st.title("Climate Change Chatbot")

    st.write("Ask me anything about climate change!")

    if "session_state" not in st.session_state:
        st.session_state.session_state = get_initial_session_state()

    input_container = st.container()
    colored_header(label='', description='', color_name='blue-30')
    response_container = st.container()

    with input_container:
        if len(st.session_state.session_state["messages"]) >= 10:
            st.write("Sorry, you've reached the limit of messages you can send. Please restart the session.")
            if st.button("Restart Session", key="restart_button", help="Click to restart the session"):
                st.session_state.session_state = get_initial_session_state()
                st.experimental_rerun()
            prompt = None
        else:
            prompt = st.chat_input("What would you like to know about climate change?")

    if prompt:
        if prompt in st.session_state.session_state["cached_responses"]:
            response = st.session_state.session_state["cached_responses"][prompt]
        else:
            context = st.session_state.session_state["context"]
            response = run_chat(prompt, context)
            st.session_state.session_state["cached_responses"][prompt] = response

        st.session_state.session_state["messages"].append({"role": "user", "content": prompt})
        st.session_state.session_state["messages"].append({"role": "assistant", "content": response})
        st.session_state.session_state["messages"] = st.session_state.session_state["messages"][-10:]
        st.session_state.session_state["context"] += f"User: {prompt}\nAssistant: {response}\n"

    with response_container:
        for i in range(len(st.session_state.session_state["messages"]) - 1, -1, -2):
            with st.chat_message("assistant"):
                response_generator = simulate_typing(st.session_state.session_state["messages"][i]["content"])
                st.write_stream(response_generator)

            if i > 0:
                with st.chat_message("user"):
                    st.markdown(st.session_state.session_state["messages"][i - 1]["content"])

if __name__ == "__main__":
    main()