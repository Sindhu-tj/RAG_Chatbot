import streamlit as st
import tempfile
import os
import json
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
# SESSION STATE
# =========================================================
if "last_file" not in st.session_state:
    st.session_state.last_file = None

if "answer" not in st.session_state:
    st.session_state.answer = ""

if "docs" not in st.session_state:
    st.session_state.docs = []

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

.block-container{
    padding-top:1.5rem;
    max-width:1100px;
}

h1{
    text-align:center;
}

.stTextInput input{
    border-radius:12px;
}

.stButton button{
    border-radius:12px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================
st.title("🤖 Universal AI File Chatbot")

st.write("""
Upload PDF, DOCX, TXT, CSV, XLSX, PPTX, JSON, XML, Markdown or Images.

Ask questions using AI-powered RAG.
""")

# =========================================================
# OCR READER
# =========================================================
@st.cache_resource
def load_ocr():

    return easyocr.Reader(
        ['en'],
        gpu=False
    )

# =========================================================
# EXTRACT TEXT
# =========================================================
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

                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)

            docs = loader.load()

            text = "\n".join([
                doc.page_content
                for doc in docs
            ])

            os.unlink(tmp_path)

        # DOCX
        elif file_type == "docx":

            doc = Document(uploaded_file)

            text = "\n".join([
                para.text
                for para in doc.paragraphs
                if para.text.strip()
            ])

        # TXT
        elif file_type == "txt":

            text = uploaded_file.read().decode(
                "utf-8",
                errors="ignore"
            )

        # JSON
        elif file_type == "json":

            data = json.load(uploaded_file)

            text = json.dumps(
                data,
                indent=2
            )

        # XML
        elif file_type == "xml":

            text = uploaded_file.read().decode(
                "utf-8",
                errors="ignore"
            )

        # MARKDOWN
        elif file_type == "md":

            text = uploaded_file.read().decode(
                "utf-8",
                errors="ignore"
            )

        # CSV
        elif file_type == "csv":

            uploaded_file.seek(0)

            df = pd.read_csv(uploaded_file)

            df = df.fillna("")

            text = df.to_string(index=False)

        # XLSX / XLS
        elif file_type in ["xlsx", "xls"]:

            uploaded_file.seek(0)

            excel_data = pd.read_excel(
                uploaded_file,
                sheet_name=None
            )

            all_text = []

            for sheet_name, df in excel_data.items():

                df = df.fillna("")

                all_text.append(
                    f"\n===== SHEET: {sheet_name} =====\n"
                )

                all_text.append(
                    df.to_string(index=False)
                )

            text = "\n".join(all_text)

        # PPTX
        elif file_type == "pptx":

            prs = Presentation(uploaded_file)

            slide_text = []

            for slide_no, slide in enumerate(prs.slides):

                slide_text.append(
                    f"\n===== SLIDE {slide_no + 1} ====="
                )

                for shape in slide.shapes:

                    if hasattr(shape, "text"):

                        if shape.text.strip():

                            slide_text.append(
                                shape.text
                            )

            text = "\n".join(slide_text)

        # IMAGE OCR
        elif file_type in ["png", "jpg", "jpeg"]:

            image = Image.open(uploaded_file)

            reader = load_ocr()

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
# VECTORSTORE
# =========================================================
@st.cache_resource
def create_vectorstore(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
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
# MODEL
# =========================================================
@st.cache_resource
def load_model():

    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_new_tokens=100,
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

    summary_questions = [
        "what is in the file",
        "summary",
        "summarize",
        "describe file",
        "about file",
        "company names",
        "which companies",
        "list company"
    ]

    # =====================================================
    # SUMMARY MODE
    # =====================================================
    if any(q in lower_question for q in summary_questions):

        lines = context.split("\n")

        clean_lines = []

        for line in lines:

            line = line.strip()

            if len(line) > 3:

                clean_lines.append(line)

        unique_lines = []

        for item in clean_lines:

            if item not in unique_lines:

                unique_lines.append(item)

        formatted = "\n".join([
            f"• {line}"
            for line in unique_lines[:20]
        ])

        answer = f"""
This file contains:

{formatted}
"""

        return answer, docs

    # =====================================================
    # NORMAL QA
    # =====================================================
    prompt = f"""
Answer the question ONLY from the context.

Rules:
- Give short accurate answers
- Do not hallucinate
- If answer not found say:
"I could not find that information in the uploaded file."

Context:
{context}

Question:
{question}

Answer:
"""

    result = model(
        prompt
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
        "xls",
        "pptx",
        "json",
        "xml",
        "md",
        "png",
        "jpg",
        "jpeg"
    ]
)

# =========================================================
# MAIN APP
# =========================================================
if uploaded_file:

    # RESET OLD RESULTS
    if st.session_state.last_file != uploaded_file.name:

        st.session_state.answer = ""
        st.session_state.docs = []
        st.session_state.last_file = uploaded_file.name

    with st.spinner("📚 Reading file..."):

        text = extract_text(uploaded_file)

    if text:

        with st.spinner("🧠 Building AI knowledge base..."):

            vectorstore = create_vectorstore(text)

            model = load_model()

        st.success("✅ File uploaded successfully!")

        file_type = uploaded_file.name.split(".")[-1].lower()

        # =================================================
        # FILE PREVIEW
        # =================================================
        with st.expander("📄 File Preview"):

            if file_type == "csv":

                uploaded_file.seek(0)

                df = pd.read_csv(uploaded_file)

                st.dataframe(df)

            elif file_type in ["xlsx", "xls"]:

                uploaded_file.seek(0)

                excel_data = pd.read_excel(
                    uploaded_file,
                    sheet_name=None
                )

                for sheet_name, df in excel_data.items():

                    st.subheader(
                        f"📑 Sheet: {sheet_name}"
                    )

                    st.dataframe(df)

            elif file_type in ["png", "jpg", "jpeg"]:

                uploaded_file.seek(0)

                image = Image.open(uploaded_file)

                st.image(
                    image,
                    use_container_width=True
                )

                st.write(text[:3000])

            elif file_type == "json":

                uploaded_file.seek(0)

                data = json.load(uploaded_file)

                st.json(data)

            else:

                st.write(text[:3000])

        # QUESTION
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

                st.session_state.answer = answer
                st.session_state.docs = docs

        # SHOW ANSWER
        if st.session_state.answer:

            st.subheader("🤖 Answer")

            st.write(st.session_state.answer)

            # SOURCE CHUNKS
            with st.expander("📌 Source Chunks Used"):

                for i, doc in enumerate(st.session_state.docs):

                    st.markdown(f"### Chunk {i+1}")

                    st.write(
                        doc.page_content[:1000]
                    )

                    st.divider()

    else:

        st.error("❌ No readable content found.")

else:

    st.info("📂 Upload a file to begin.")