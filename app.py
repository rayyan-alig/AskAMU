import streamlit as st
import os
from dotenv import load_dotenv
from pathlib import Path

# --- Load password from environment or Streamlit secrets ---
load_dotenv() # Load .env for local development
CORRECT_PASSWORD = os.getenv("APP_PASSWORD") or st.secrets.get("auth", {}).get("password", None)

# --- Authentication ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.set_page_config(page_title="Login", page_icon="🔐", layout="centered")
    st.title("🔐 Secure Login for AskAMU, an AI Assistant for AMU")
    password = st.text_input("The App is password protected to avoid the api limits, DM me to get the password", type="password", placeholder="Enter app password")

    if st.button("Login"):
        if password == CORRECT_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ Incorrect password. Try again.")
    st.stop()


from rag_pipeline import load_index
# ✨ Import the more robust chat engine
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import Settings


# --- Custom CSS for Mobile View ---
st.markdown("""
<style>
div[data-testid="stVerticalBlock"] {
    padding-bottom: 4rem;
}
</style>
""", unsafe_allow_html=True)

# --- Streamlit App Config ---
st.set_page_config(
    page_title="AMU AI Assistant",
    page_icon="🎓",
    layout="centered"
)
st.markdown(
    """
    <h1 style='text-align: center; font-family: Georgia, serif;'>𓂃🖊 AskAMU</h1>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <p style='text-align: center; color: gray; font-size: 0.9em;'>
        Your friendly AI assistant for AMU related questions.
    </p>
    """,
    unsafe_allow_html=True
)


# --- Initialize Session State Memory ---
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ChatMemoryBuffer.from_defaults(token_limit=5000)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you today?"}
    ]



# --- Load RAG Chat Engine ---
@st.cache_resource
def get_chat_engine():
    try:
        index = load_index()

        system_prompt = """
        You are an AI assistant for Aligarh Muslim University (AMU).
        You must answer ONLY using the retrieved documents from the vector database(retriever).

        🔒 RULES:
        1. You MAY use past user questions to understand intent, but you MUST answer ONLY using retrieved documents.
        2. If the answer is NOT found in the retrieved context, respond:
        "I'm sorry, I don't have information on that topic based on the provided documents."
        3. Do NOT use prior knowledge, assumptions, or anything from memory.
        4. Do NOT generate answers based on previous chat history.
        5. Keep your answers factual, concise, and grounded in the retrieved context only.
        """
        
        chat_engine = CondensePlusContextChatEngine.from_defaults(
            retriever=index.as_retriever(similarity_top_k=10), 
            memory=st.session_state.chat_memory,
            system_prompt=system_prompt,
            llm=Settings.llm,
        )
        return chat_engine
    except Exception as e:
        st.error(f"❌ Failed to load chat engine: {e}")
        return None


chat_engine = get_chat_engine()

# --- Sidebar for Logout ---
with st.sidebar:
    if st.button("🔓 Logout"):
        st.session_state.authenticated = False
        st.rerun()

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --- Handle User Input ---
if prompt := st.chat_input("Ask me anything about AMU..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            try:
                if chat_engine:
                    # Use the chat engine's stream_chat method directly
                    streaming_response = chat_engine.stream_chat(prompt)
                    full_response = ""
                    for token in streaming_response.response_gen:
                        full_response += token
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    
                    source_nodes = getattr(streaming_response, 'source_nodes', [])
                    if source_nodes:
                        sources = list(set(node.metadata.get("file_name") or node.metadata.get("source_url") for node in source_nodes if node.metadata))
                        if sources:
                            with st.expander("📚 Sources"):
                                for src in sources[:3]: 
                                    if src.startswith("http"):
                                        st.markdown(f"- [🔗 {src.split('/')[-1]}]({src})")
                                    else:
                                        st.markdown(f"- 📄 `{src}`")
                else:
                    full_response = "⚠️ Chat engine not loaded. Please try again later."
                    message_placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"❌ Error during query: {e}"
                message_placeholder.markdown(full_response)

    # Save assistant reply to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    
    



