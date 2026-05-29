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
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.stDataFrame {
    border-radius: 12px;
}

.stButton > button {
    border-radius: 10px;
    width: 220px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================
if "last_file" not in st.session_state:
    st.session_state.last_file = None

if "answer" not in st.session_state:
    st.session_state.answer = ""

if "docs" not in st.session_state:
    st.session_state.docs = []

if "df" not in st.session_state:
    st.session_state.df = None


# =========================================================
# OCR
# =========================================================
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)


# =========================================================
# MODEL
# =========================================================
@st.cache_resource
def load_model():

    return pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        max_new_tokens=256,
        do_sample=False,
        temperature=0
    )


# =========================================================
# TEXT EXTRACTION
# =========================================================
def extract_text(uploaded_file):

    file_type = uploaded_file.name.split(".")[-1].lower()

    text = ""

    try:

        # =====================================================
        # PDF
        # =====================================================
        if file_type == "pdf":

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:

                tmp.write(uploaded_file.read())
                path = tmp.name

            loader = PyPDFLoader(path)

            docs = loader.load()

            text = "\n".join(
                [d.page_content for d in docs]
            )

            os.unlink(path)

        # =====================================================
        # DOCX
        # =====================================================
        elif file_type == "docx":

            doc = Document(uploaded_file)

            text = "\n".join(
                [
                    p.text
                    for p in doc.paragraphs
                    if p.text.strip()
                ]
            )

        # =====================================================
        # TXT
        # =====================================================
        elif file_type == "txt":

            text = uploaded_file.read().decode(
                "utf-8",
                errors="ignore"
            )

        # =====================================================
        # JSON / XML / MD
        # =====================================================
        elif file_type in ["json", "xml", "md"]:

            text = uploaded_file.read().decode(
                "utf-8",
                errors="ignore"
            )

        # =====================================================
        # CSV
        # =====================================================
        elif file_type == "csv":

            df = pd.read_csv(uploaded_file).fillna("")

            st.session_state.df = df

            text = df.to_string(index=False)

        # =====================================================
        # EXCEL
        # =====================================================
        elif file_type in ["xlsx", "xls"]:

            df = pd.read_excel(uploaded_file).fillna("")

            st.session_state.df = df

            text = df.to_string(index=False)

        # =====================================================
        # PPTX
        # =====================================================
        elif file_type == "pptx":

            prs = Presentation(uploaded_file)

            slides = []

            for i, slide in enumerate(prs.slides):

                slides.append(f"\nSLIDE {i+1}")

                for shape in slide.shapes:

                    if hasattr(shape, "text"):

                        if shape.text.strip():

                            slides.append(shape.text)

            text = "\n".join(slides)

        # =====================================================
        # IMAGE OCR
        # =====================================================
        elif file_type in ["png", "jpg", "jpeg"]:

            image = Image.open(uploaded_file)

            reader = load_ocr()

            results = reader.readtext(image)

            text = "\n".join(
                [r[1] for r in results]
            )

            text = " ".join(text.split())

        return text.strip()

    except Exception as e:

        st.error(f"Error while reading file: {e}")

        return ""


# =========================================================
# VECTOR DATABASE
# =========================================================
@st.cache_resource
def create_vectorstore(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )

    chunks = splitter.split_text(text)

    docs = []

    for i, c in enumerate(chunks):

        docs.append(
            LangDocument(
                page_content=c,
                metadata={
                    "chunk_id": i
                }
            )
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5"
    )

    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    return vectorstore


# =========================================================
# SMART ANSWER ENGINE
# =========================================================
def smart_answer(
    file_type,
    question,
    text,
    vectorstore,
    model
):

    q = question.lower()

    # =====================================================
    # CSV / EXCEL
    # =====================================================
    if file_type in ["csv", "xlsx", "xls"]:

        df = st.session_state.df

        # SUMMARY
        if any(word in q for word in [
            "summary",
            "overview",
            "describe",
            "what",
            "file"
        ]):

            return {
                "type": "table_summary",
                "rows": df.shape[0],
                "cols": df.shape[1],
                "columns": list(df.columns),
                "preview": df.head(20)
            }

        # COLUMNS
        elif "column" in q:

            return {
                "type": "text",
                "content":
                f"Columns are:\n\n{', '.join(df.columns)}"
            }

        # ROW COUNT
        elif "rows" in q or "entries" in q:

            return {
                "type": "text",
                "content":
                f"The file contains {df.shape[0]} rows."
            }

        # SEARCH DATA
        else:

            matched = pd.DataFrame()

            for col in df.columns:

                temp = df[
                    df[col]
                    .astype(str)
                    .str.lower()
                    .str.contains(q, na=False)
                ]

                matched = pd.concat([matched, temp])

            matched = matched.drop_duplicates()

            if len(matched) > 0:

                return {
                    "type": "table",
                    "data": matched.head(50)
                }

            return {
                "type": "text",
                "content":
                "No matching data found."
            }

    # =====================================================
    # JSON
    # =====================================================
    elif file_type == "json":

        try:

            data = json.loads(text)

            return {
                "type": "text",
                "content":
                f"JSON contains keys:\n\n{list(data.keys())[:20]}"
            }

        except:

            return {
                "type": "text",
                "content":
                "Invalid JSON structure."
            }

    # =====================================================
    # IMAGE OCR
    # =====================================================
    elif file_type in ["png", "jpg", "jpeg"]:

        return {
            "type": "text",
            "content":
            f"Extracted text from image:\n\n{text[:3000]}"
        }

    # =====================================================
    # PPTX SUMMARY
    # =====================================================
    elif file_type == "pptx":

        prompt = f"""
Summarize this PowerPoint presentation clearly.

TEXT:
{text[:4000]}
"""

        result = model(prompt)[0]["generated_text"]

        return {
            "type": "text",
            "content": result
        }

    # =====================================================
    # GENERAL RAG MODE
    # =====================================================
    else:

        docs = vectorstore.similarity_search(
            question,
            k=4
        )

        context = "\n".join(
            [d.page_content for d in docs]
        )

        prompt = f"""
You are a highly accurate document assistant.

RULES:
- Answer ONLY from provided context
- If answer is missing say:
  'Information not found in document'
- Never hallucinate
- Be concise and factual

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

        result = model(prompt)[0]["generated_text"]

        return {
            "type": "rag",
            "content": result,
            "docs": docs
        }


# =========================================================
# MAIN UI
# =========================================================
st.title("🤖 Universal AI File Chatbot")

uploaded_file = st.file_uploader(
    "Upload File",
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
# FILE LOADED
# =========================================================
if uploaded_file:

    file_type = uploaded_file.name.split(".")[-1].lower()

    # RESET
    if st.session_state.last_file != uploaded_file.name:

        st.session_state.answer = ""
        st.session_state.docs = []
        st.session_state.df = None

        st.session_state.last_file = uploaded_file.name

    # READ FILE
    with st.spinner("Reading file..."):

        text = extract_text(uploaded_file)

    if text:

        # CREATE VECTORSTORE
        with st.spinner("Creating AI index..."):

            vectorstore = create_vectorstore(text)

            model = load_model()

        # =====================================================
        # SIDEBAR
        # =====================================================
        with st.sidebar:

            st.markdown("## 📁 Uploaded File")

            st.success("File uploaded successfully!")

            st.markdown("## 📄 File Information")

            st.write(f"**File Name:** {uploaded_file.name}")

            st.write(f"**File Type:** {file_type.upper()}")

            if st.session_state.df is not None:

                df = st.session_state.df

                st.write(f"**Rows:** {df.shape[0]}")

                st.write(f"**Columns:** {df.shape[1]}")

                st.write(
                    f"**Column Names:** "
                    f"{', '.join(df.columns)}"
                )

        # =====================================================
        # PREVIEW
        # =====================================================
        st.subheader("📄 File Preview")

        if file_type in ["csv", "xlsx", "xls"]:

            st.dataframe(
                st.session_state.df.head(20),
                use_container_width=True,
                height=500
            )

        else:

            st.text(text[:3000])

        # =====================================================
        # QUESTION
        # =====================================================
        question = st.text_input(
            "Ask a question"
        )

        # =====================================================
        # AI ANSWER
        # =====================================================
        if question:

            with st.spinner("Thinking..."):

                response = smart_answer(
                    file_type,
                    question,
                    text,
                    vectorstore,
                    model
                )

            st.subheader("🤖 AI Answer")

            # =================================================
            # TABLE SUMMARY
            # =================================================
            if response["type"] == "table_summary":

                st.success(
                    "File analyzed successfully."
                )

                st.markdown(f"""
- **Rows:** {response['rows']}
- **Columns:** {response['cols']}
- **Column Names:** {', '.join(response['columns'])}
""")

                st.dataframe(
                    response["preview"],
                    use_container_width=True,
                    height=500
                )

            # =================================================
            # TABLE
            # =================================================
            elif response["type"] == "table":

                st.dataframe(
                    response["data"],
                    use_container_width=True,
                    height=500
                )

            # =================================================
            # TEXT
            # =================================================
            elif response["type"] == "text":

                st.write(response["content"])

            # =================================================
            # RAG
            # =================================================
            elif response["type"] == "rag":

                st.write(response["content"])

                with st.expander("📌 Source Details"):

                    st.write(
                        f"{len(response['docs'])} "
                        f"relevant chunks retrieved."
                    )

                    for i, d in enumerate(response["docs"]):

                        st.markdown(
                            f"### Chunk {i+1}"
                        )

                        st.write(
                            d.page_content[:800]
                        )

    else:

        st.error("No content found.")

# =========================================================
# NO FILE
# =========================================================
else:

    st.info("Upload a file to start.")

