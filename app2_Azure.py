import streamlit as st
import random
import time
import asyncio
from llm_model2_azure import run_workflow  # 'run_workflow' is the async function in llm_model2.py

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

async def run_chat(question, context):
    inputs = {"question": question, "context": context}
    response_text = ""
    citations = ""
    try:
        result = await run_workflow(inputs)
        response_text = result["generation"]
        citations = result["citations"]
    except Exception as e:
        response_text = str(e)
    return response_text, citations

def simulate_typing(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

def main():
    st.set_page_config(page_icon=None, page_title="Multilingual Climate Chatbot", layout="centered", initial_sidebar_state="auto", menu_items=None)

    with st.sidebar:
        st.title('Multilingual Climate Chatbot')
        st.markdown('''
            ## About
            This app is an AI-powered chatbot built using:
            - Streamlit
            - Command R- Plus AI model
            - The purpose of this app is to educate individuals about climate change and foster a community of informed citizens. It provides accurate information and resources about climate change and its impacts, and encourages users to take action in their own communities.
        ''')
        add_vertical_space(5)
        st.markdown('<div>Made by:</div>', unsafe_allow_html=True)
        st.image("tree.ico", width=40)
        st.markdown('<div style="font-size: 18px;">Climate Resilient Communities</div>', unsafe_allow_html=True)



    col1, col2 = st.columns([1, 8])

    with col1:
        add_vertical_space(1)
        st.image("CCCicon.png", width=80)  # Adjust the size here
        add_vertical_space(4)  # Adjust spacing to move image down

    with col2:
        st.title("Multilingual Climate Chatbot")
        st.write("Ask me anything about climate change!")

    if "session_state" not in st.session_state:
        st.session_state.session_state = get_initial_session_state()

    response_container = st.container()

    with response_container:
        for i in range(len(st.session_state.session_state["messages"]) - 1, -1, -2):
            with st.chat_message("assistant"):
                response = st.session_state.session_state["messages"][i]["content"]
                citations = st.session_state.session_state["messages"][i].get("citations", "")
                st.markdown(response)
                if citations:
                    st.markdown(f"**Sources:**  \n {citations}")

            if i > 0:
                with st.chat_message("user"):
                    st.markdown(st.session_state.session_state["messages"][i - 1]["content"])

    colored_header(label='', description='', color_name='blue-30')

    if len(st.session_state.session_state["messages"]) < 10:
        prompt = st.chat_input("What would you like to know about climate change?")

        if prompt:
            if prompt in st.session_state.session_state["cached_responses"]:
                response, citations = st.session_state.session_state["cached_responses"][prompt]
            else:
                context = st.session_state.session_state["context"]
                response, citations = asyncio.run(run_chat(prompt, context))
                st.session_state.session_state["cached_responses"][prompt] = (response, citations)

            st.session_state.session_state["messages"].append({"role": "user", "content": prompt})
            st.session_state.session_state["messages"].append({"role": "assistant", "content": response, "citations": citations})
            st.session_state.session_state["messages"] = st.session_state.session_state["messages"][-10:]
            st.session_state.session_state["context"] += f"User: {prompt}\nAssistant: {response}\n"
            st.rerun()  # Rerun to update the display

if __name__ == "__main__":
    main()