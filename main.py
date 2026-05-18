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
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="wide"
)


# ==========================================
# LOAD PDF + CREATE VECTOR DATABASE
# ==========================================
def load_and_index(pdf_file):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
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
        max_new_tokens=256,
        temperature=0.3
    )

    llm = HuggingFacePipeline(
        pipeline=pipe
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    prompt = PromptTemplate.from_template(
        """
You are an intelligent AI assistant.

Use ONLY the provided PDF context to answer the question clearly and professionally.

If the answer is not available in the context, say:
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
            doc.page_content for doc in docs
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

    answer = chain.invoke(question)
    source_docs = retriever.invoke(question)

    return answer, source_docs


# ==========================================
# STREAMLIT UI
# ==========================================
st.title("📄 PDF RAG Chatbot")
st.write("Upload a PDF and ask questions from it.")

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type="pdf"
)

if uploaded_file:

    with st.spinner("Processing PDF..."):
        vectorstore = load_and_index(uploaded_file)
        chain, retriever = build_qa_chain(vectorstore)

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

        with st.expander("View Source Context"):
            for doc in source_docs:
                st.write(doc.page_content)
                st.divider()