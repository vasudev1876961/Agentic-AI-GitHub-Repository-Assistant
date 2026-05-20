import logging

import streamlit as st
from google import genai
from prompt_flow import execute_rag_flow

from coderag.config import GEMINI_API_KEY, GEMINI_CHAT_MODEL


# Logging Configuration

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


# Gemini Initialization

client = None
try:
    if not GEMINI_API_KEY:
        raise ValueError("Gemini API key not found in environment variables")

    client = genai.Client(api_key=GEMINI_API_KEY)
    logger.info(f" Gemini client initialized successfully ({GEMINI_CHAT_MODEL})")

except Exception as e:
    logger.error(f" Failed to initialize Gemini client: {e}")


# Streamlit Setup

st.set_page_config(page_title="GitRepo: Agentic Assistant", page_icon="", layout="wide")

st.title(" GitRepo: Agentic Coding Assistant ")
st.markdown("*AI-powered RAG + Gemini reasoning with tool usage*")


# Session State

if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_context" not in st.session_state:
    st.session_state.conversation_context = []


# Sidebar Controls

with st.sidebar:
    st.header(" Controls")

    st.subheader(" Load a GitHub Repo")
    repo_url = st.text_input(
        "Paste GitHub URL", placeholder="https://github.com/owner/repo.git"
    )

    colA, colB = st.columns(2)
    do_clean = colA.checkbox("Reclone if exists", value=True)
    do_index = colB.checkbox("Index after clone", value=True)

    if st.button("Clone & Index"):
        if not repo_url.strip():
            st.error("Please paste a valid GitHub URL.")
        else:
            with st.spinner("Cloning repository..."):
                try:
                    from ingestor import index_directory
                    from repo_loader import clone_repo

                    repo_dir, repo_name = clone_repo(
                        repo_url.strip(), clean_if_exists=do_clean
                    )
                    st.success(f" Cloned to: {repo_dir}")

                    if do_index:
                        with st.spinner("Indexing repository..."):
                            n = index_directory(repo_dir)
                        st.success(f"Indexed {n} files from '{repo_name}'.")
                except Exception as e:
                    st.error(f" Failed to clone or index repository: {e}")

    st.divider()

    if st.button(" Clear Conversation", type="secondary"):
        st.session_state.messages = []
        st.session_state.conversation_context = []
        st.rerun()

    st.header(" Status")
    if client:
        st.success(" Gemini Connected")
    else:
        st.error(" Gemini Not Connected. Check your GEMINI_API_KEY in the .env file.")

    if st.session_state.messages:
        st.info(f" {len(st.session_state.messages)} messages so far")


# Display Chat History

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat Logic

if not client:
    st.warning(" Gemini client not available. Please configure your API key.")
    st.stop()

if prompt := st.chat_input("Ask your coding question...", disabled=not client):
    if not prompt.strip():
        st.warning("Please enter a valid question.")
        st.stop()

    # Save user query
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.conversation_context.append(f"User: {prompt}")

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        with st.spinner(" Analyzing your codebase and generating a response..."):
            try:
                # Step 1️ Retrieve relevant code snippets (RAG)
                retrieved_context = execute_rag_flow(prompt)

                # Step 2️ Pass context + question to Agent (Gemini)
                from coderag.agent import run_agent

                answer = run_agent(prompt)  # ONLY user query goes to planner

                if not answer or not answer.strip():
                    answer = " No response from Gemini."

                message_placeholder.markdown(answer)
                full_response = answer

            except Exception as e:
                error_message = f" Error during processing: {str(e)}"
                logger.error(error_message)
                message_placeholder.error(error_message)
                full_response = error_message

        # Step 3️ Store assistant response
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
        st.session_state.conversation_context.append(
            f"Assistant: {full_response[:300]}..."
        )

        # Keep context size manageable
        if len(st.session_state.conversation_context) > 20:
            st.session_state.conversation_context = (
                st.session_state.conversation_context[-20:]
            )


# Footer Section

if not st.session_state.messages:
    st.markdown("---")
    st.markdown("###  Tips for Best Results:")
    st.markdown("""
        - Ask detailed questions about your code
        - Mention function or class names for precision
        - Try: “Explain how FAISS indexing works”
        - Try: “Summarize what embeddings are used here”
        """)
