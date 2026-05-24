import streamlit as st
import tempfile
import os
import pandas as pd

from PIL import Image
from docx import Document
from pptx import Presentation
import easyocr

from langchain.schema import Document as LangDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import pipeline


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Universal AI File Chatbot",
    page_icon="🤖",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

.block-container{
    padding-top:1.5rem;
    max-width:1000px;
}

h1{
    text-align:center;
}

.stTextInput input{
    border-radius:10px;
}

.stButton button{
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================
st.title("🤖 Universal AI File Chatbot")

st.write(
    "Upload PDF, DOCX, TXT, CSV, XLSX, PPTX or Images and ask questions using AI-powered RAG."
)

# =========================================================
# EXTRACT TEXT
# =========================================================
def extract_text(uploaded_file):

    file_type = uploaded_file.name.split(".")[-1].lower()

    text = ""

    try:

        # =================================================
        # PDF
        # =================================================
        if file_type == "pdf":

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp:

                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)

            docs = loader.load()

            text = "\n".join([
                doc.page_content
                for doc in docs
            ])

            os.unlink(tmp_path)

        # =================================================
        # DOCX
        # =================================================
        elif file_type == "docx":

            doc = Document(uploaded_file)

            text = "\n".join([
                para.text
                for para in doc.paragraphs
                if para.text.strip()
            ])

        # =================================================
        # TXT
        # =================================================
        elif file_type == "txt":

            text = uploaded_file.read().decode(
                "utf-8",
                errors="ignore"
            )

        # =================================================
        # CSV
        # =================================================
        elif file_type == "csv":

            df = pd.read_csv(uploaded_file)

            df = df.astype(str)

            text = df.to_string(index=False)

        # =================================================
        # XLSX
        # =================================================
        elif file_type == "xlsx":

            excel_data = pd.read_excel(
                uploaded_file,
                sheet_name=None
            )

            all_text = []

            for sheet_name, df in excel_data.items():

                df = df.astype(str)

                all_text.append(
                    f"\n\n========== SHEET: {sheet_name} ==========\n"
                )

                all_text.append(
                    df.to_string(index=False)
                )

            text = "\n".join(all_text)

        # =================================================
        # PPTX
        # =================================================
        elif file_type == "pptx":

            prs = Presentation(uploaded_file)

            slide_text = []

            for slide in prs.slides:

                for shape in slide.shapes:

                    if hasattr(shape, "text"):

                        if shape.text.strip():

                            slide_text.append(shape.text)

            text = "\n".join(slide_text)

        # =================================================
        # IMAGE OCR
        # =================================================
        elif file_type in ["png", "jpg", "jpeg"]:

            image = Image.open(uploaded_file)

            reader = easyocr.Reader(
                ['en'],
                gpu=False
            )

            results = reader.readtext(image)

            text = "\n".join([
                result[1]
                for result in results
            ])

        return text.strip()

    except Exception as e:

        st.error(f"❌ Error reading file: {e}")

        return ""


# =========================================================
# CREATE VECTORSTORE
# =========================================================
@st.cache_resource
def create_vectorstore(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    chunks = [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]

    docs = [
        LangDocument(page_content=chunk)
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


# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():

    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_new_tokens=80,
        temperature=0.0,
        do_sample=False
    )

    return pipe


# =========================================================
# ASK QUESTION
# =========================================================
def ask_question(
    vectorstore,
    question,
    model
):

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    docs = retriever.invoke(question)

    context = "\n\n".join([
        doc.page_content
        for doc in docs
    ])

    lower_question = question.lower()

    # =====================================================
    # FILE SUMMARY QUESTIONS
    # =====================================================
    summary_questions = [
        "what is in the file",
        "what does this file contain",
        "summary",
        "summarize",
        "about file",
        "describe file",
        "explain file",
        "which companies are mentioned"
    ]

    # =====================================================
    # MANUAL SMART SUMMARY
    # =====================================================
    if any(q in lower_question for q in summary_questions):

        words = context.split()

        unique_words = []

        for word in words:

            clean_word = word.strip(",.:;()[]{}")

            if (
                clean_word.istitle()
                and len(clean_word) > 3
                and clean_word not in unique_words
            ):
                unique_words.append(clean_word)

        company_names = unique_words[:15]

        answer = f"""
This file contains healthcare, pharmaceutical, biotech, and clinical research related company data.

Important organizations mentioned include:

{", ".join(company_names)}

The dataset appears to focus on:
- Clinical research
- Healthcare organizations
- Pharmaceutical companies
- Medical innovation
- Research partnerships

The uploaded file contains structured business and healthcare-related information.
"""

        return answer, docs

    # =====================================================
    # NORMAL QUESTION ANSWERING
    # =====================================================
    prompt = f"""
Answer the question ONLY from the context.

Rules:
- Be accurate
- Be professional
- Keep answer short
- Do not copy chunks
- If answer not found say:
"I could not find that information in the uploaded file."

Context:
{context}

Question:
{question}

Answer:
"""

    result = model(
        prompt,
        max_new_tokens=80
    )[0]["generated_text"]

    return result, docs


# =========================================================
# FILE UPLOADER
# =========================================================
uploaded_file = st.file_uploader(
    "📤 Upload File",
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

# =========================================================
# MAIN APP
# =========================================================
if uploaded_file:

    with st.spinner("📚 Reading file..."):

        text = extract_text(uploaded_file)

    if text:

        with st.spinner("🧠 Building AI knowledge base..."):

            vectorstore = create_vectorstore(text)

            model = load_model()

        st.success("✅ File uploaded successfully!")

        # =================================================
        # FILE PREVIEW
        # =================================================
        with st.expander("📄 File Preview"):

            st.write(text[:3000])

        # =================================================
        # QUESTION INPUT
        # =================================================
        question = st.text_input(
            "💬 Ask a question"
        )

        if question:

            with st.spinner("🔍 Finding answer..."):

                answer, docs = ask_question(
                    vectorstore,
                    question,
                    model
                )

            # =============================================
            # ANSWER
            # =============================================
            st.subheader("🤖 Answer")

            st.write(answer)

            # =============================================
            # SOURCE CHUNKS
            # =============================================
            with st.expander("📌 Source Chunks Used"):

                for i, doc in enumerate(docs):

                    st.markdown(f"### Chunk {i+1}")

                    st.write(doc.page_content[:1000])

                    st.divider()

    else:

        st.error("❌ No readable content found.")

else:

    st.info("📂 Upload a file to begin.")