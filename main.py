import streamlit as st
import tempfile
import os
import pandas as pd
import easyocr
from PIL import Image
from docx import Document
from pptx import Presentation

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangDocument
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import pipeline


# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Universal AI File Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ==========================================
# UI
# ==========================================
st.title("🤖 Universal AI File Chatbot")

st.write(
    "Upload PDF, DOCX, TXT, CSV, XLSX, PPTX or Image and ask questions."
)


# ==========================================
# EXTRACT TEXT
# ==========================================
def extract_text(uploaded_file):

    file_type = uploaded_file.name.split(".")[-1].lower()

    text = ""

    try:

        # PDF
        if file_type == "pdf":

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp:

                tmp.write(
                    uploaded_file.read()
                )

                tmp_path = tmp.name

            loader = PyPDFLoader(
                tmp_path
            )

            docs = loader.load()

            text = "\n".join(
                [
                    doc.page_content
                    for doc in docs
                ]
            )

            os.unlink(tmp_path)

        # DOCX
        elif file_type == "docx":

            doc = Document(
                uploaded_file
            )

            text = "\n".join(
                [
                    para.text
                    for para in doc.paragraphs
                    if para.text.strip()
                ]
            )

        # TXT
        elif file_type == "txt":

            text = uploaded_file.read().decode(
                "utf-8",
                errors="ignore"
            )

        # CSV
        elif file_type == "csv":

            df = pd.read_csv(
                uploaded_file
            )

            text = df.to_string()

        # XLSX
        elif file_type == "xlsx":

            df = pd.read_excel(
                uploaded_file
            )

            text = df.to_string()

        # PPTX
        elif file_type == "pptx":

            prs = Presentation(
                uploaded_file
            )

            slide_text = []

            for slide in prs.slides:

                for shape in slide.shapes:

                    if hasattr(
                        shape,
                        "text"
                    ):

                        if shape.text.strip():

                            slide_text.append(
                                shape.text
                            )

            text = "\n".join(
                slide_text
            )

        # IMAGE OCR
        elif file_type in [
            "png",
            "jpg",
            "jpeg"
        ]:

            image = Image.open(
                uploaded_file
            )

            reader = easyocr.Reader(
                ['en'],
                gpu=False
            )

            results = reader.readtext(
                image
            )

            text = "\n".join(
                [
                    result[1]
                    for result in results
                ]
            )

        return text.strip()

    except Exception as e:

        st.error(
            f"Error reading file: {str(e)}"
        )

        return ""


# ==========================================
# CREATE VECTORSTORE
# ==========================================
@st.cache_resource
def create_vectorstore(text):

    if not text:

        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_text(
        text
    )

    chunks = [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]

    if len(chunks) == 0:

        return None

    docs = [
        LangDocument(
            page_content=chunk
        )
        for chunk in chunks
    ]

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    return vectorstore


# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():

    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=200,
        do_sample=True,
        temperature=0.1
    )

    return pipe


# ==========================================
# ASK QUESTION
# ==========================================
def ask_question(
    vectorstore,
    question,
    model
):

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    docs = retriever.invoke(
        question
    )

    context = "\n\n".join(
        [
            doc.page_content
            for doc in docs
        ]
    )

    prompt = f"""
You are an intelligent assistant.

Answer ONLY using the uploaded file content.

Rules:
1. Do not guess.
2. Do not hallucinate.
3. If answer is unavailable say:
"I could not find that information in the uploaded file."

Context:
{context}

Question:
{question}

Answer:
"""

    response = model(
        prompt
    )[0]["generated_text"]

    return response, docs


# ==========================================
# FILE UPLOADER
# ==========================================
uploaded_file = st.file_uploader(
    "Upload File",
    type=[
        "pdf",
        "docx",
        "txt",
        "csv",
        "xlsx",
        "pptx",
        "png",
        "jpg",
        "jpeg"
    ]
)

# ==========================================
# MAIN APP
# ==========================================
if uploaded_file:

    with st.spinner(
        "Reading file..."
    ):

        text = extract_text(
            uploaded_file
        )

    if not text:

        st.error(
            "No readable content found in file."
        )

    else:

        with st.spinner(
            "Creating AI knowledge base..."
        ):

            vectorstore = create_vectorstore(
                text
            )

            model = load_model()

        st.success(
            "File uploaded successfully!"
        )

        question = st.text_input(
            "Ask a question"
        )

        if question:

            with st.spinner(
                "Generating answer..."
            ):

                answer, docs = ask_question(
                    vectorstore,
                    question,
                    model
                )

            st.subheader(
                "Answer"
            )

            st.write(
                answer
            )

            with st.expander(
                "📌 Source Chunks Used"
            ):

                for i, doc in enumerate(
                    docs
                ):

                    st.markdown(
                        f"### Chunk {i+1}"
                    )

                    st.write(
                        doc.page_content
                    )

                    st.divider()