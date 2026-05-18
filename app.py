import streamlit as st
from main import load_and_index, build_qa_chain, ask_question

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI RAG PDF Chatbot",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.stChatMessage {
    border-radius: 12px;
    padding: 10px;
}

h1 {
    color: #2563eb;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🤖 AI PDF Assistant")
st.sidebar.write(
    "Upload one or more PDFs and ask questions using AI-powered semantic search."
)

st.sidebar.markdown("### 🚀 Features")
st.sidebar.markdown("""
- Multiple PDF Upload
- Semantic Search
- RAG Pipeline
- Conversational AI
- Source Chunk Retrieval
""")

# -----------------------------
# Main Title
# -----------------------------
st.title("📄 AI-Powered RAG PDF Chatbot")

st.write("""
Ask questions from your PDFs using
**LangChain, FAISS, Hugging Face Transformers, and RAG architecture**.
""")

# -----------------------------
# PDF Upload
# -----------------------------
pdf_files = st.file_uploader(
    "📤 Upload PDF Files",
    type="pdf",
    accept_multiple_files=True
)

# -----------------------------
# Process PDFs
# -----------------------------
if pdf_files:

    if "chain" not in st.session_state:

        with st.spinner("📚 Reading PDFs and building AI knowledge base..."):

            try:
                vectorstores = []

                for pdf_file in pdf_files:
                    vectorstore = load_and_index(pdf_file)
                    vectorstores.append(vectorstore)

                # Use first vectorstore for now
                # (later you can merge vectorstores)
                chain, retriever = build_qa_chain(vectorstores[0])

                st.session_state.chain = chain
                st.session_state.retriever = retriever
                st.session_state.messages = []

            except Exception as e:
                st.error(f"❌ Error loading PDFs: {e}")
                st.stop()

        st.success("✅ PDFs processed successfully!")

    # -----------------------------
    # Display Chat History
    # -----------------------------
    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # -----------------------------
    # User Question Input
    # -----------------------------
    question = st.chat_input(
        "💬 Ask a question about your PDFs..."
    )

    # -----------------------------
    # Generate Response
    # -----------------------------
    if question:

        # User Message
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.write(question)

        # Assistant Response
        with st.chat_message("assistant"):

            with st.spinner("🔍 Searching documents and generating answer..."):

                try:
                    answer, sources = ask_question(
                        st.session_state.chain,
                        st.session_state.retriever,
                        question
                    )

                except Exception as e:
                    answer = f"❌ Error: {e}"
                    sources = []

            st.write(answer)

            # -----------------------------
            # Source Chunks
            # -----------------------------
            if sources:

                with st.expander("📌 Source Chunks Used"):

                    for i, doc in enumerate(sources):

                        page = doc.metadata.get("page", "?")

                        st.markdown(
                            f"### Chunk {i+1} — Page {page}"
                        )

                        st.write(
                            doc.page_content[:300] + "..."
                        )

        # Save Assistant Response
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

# -----------------------------
# No PDF Uploaded
# -----------------------------
else:

    st.info("📂 Please upload one or more PDF files to begin.")