import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from transformers import pipeline


# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="AI RAG PDF Chatbot",
    page_icon="📄",
    layout="wide"
)


# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

h1 {
    color: white;
    text-align: center;
    font-size: 50px;
}

.stTextInput input {
    background-color: #1e293b;
    color: white;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# TITLE
# ==========================================
st.markdown("# 📄 AI-Powered RAG PDF Chatbot")

st.write(
    "Upload a PDF and ask questions based only on the PDF."
)


# ==========================================
# LOAD PDF + VECTOR DB
# ==========================================
def load_and_index(pdf_file):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    os.unlink(tmp_path)

    # Better chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore


# ==========================================
# BUILD QA CHAIN
# ==========================================
def build_qa_chain(vectorstore):

    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=120,
        temperature=0.1
    )

    llm = HuggingFacePipeline(
        pipeline=pipe
    )

    # Better retrieval
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    prompt = PromptTemplate.from_template(
        """
You are an intelligent PDF assistant.

You MUST answer ONLY using the provided context.

Rules:
1. Never guess.
2. Never add outside information.
3. Give short and precise answers.
4. For projects → list project names only.
5. For skills → list only skills.
6. For certifications → list only certifications.
7. If answer is missing, say exactly:
"I could not find that information in the PDF."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    def format_docs(docs):
        return "\n\n".join(
            [doc.page_content for doc in docs]
        )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


# ==========================================
# ASK QUESTION
# ==========================================
def ask_question(chain, retriever, question):

    source_docs = retriever.invoke(question)

    answer = chain.invoke(question)

    return answer, source_docs


# ==========================================
# FILE UPLOAD
# ==========================================
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type="pdf"
)


# ==========================================
# PROCESS PDF
# ==========================================
if uploaded_file:

    with st.spinner("Reading PDF and building AI knowledge base..."):

        vectorstore = load_and_index(uploaded_file)

        chain, retriever = build_qa_chain(
            vectorstore
        )

    st.success("PDF uploaded successfully!")

    question = st.text_input(
        "Ask a question about the PDF"
    )

    if question:

        with st.spinner("Generating answer..."):

            answer, source_docs = ask_question(
                chain,
                retriever,
                question
            )

        st.subheader("Answer")
        st.write(answer)

        with st.expander(
            "📌 Source Chunks Used"
        ):

            for i, doc in enumerate(source_docs):

                st.markdown(
                    f"### Chunk {i+1}"
                )

                st.write(
                    doc.page_content
                )

                st.divider()