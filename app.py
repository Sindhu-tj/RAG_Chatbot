import streamlit as st
import tempfile
import os
import json
import pandas as pd
import xml.etree.ElementTree as ET

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
if "answer" not in st.session_state:
    st.session_state.answer = ""

if "docs" not in st.session_state:
    st.session_state.docs = []

if "last_file" not in st.session_state:
    st.session_state.last_file = None

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

.block-container{
    padding-top:1rem;
    max-width:1100px;
}

h1{
    text-align:center;
}

.stButton button{
    border-radius:12px;
}

.stTextInput input{
    border-radius:12px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================
st.title("🤖 Universal AI File Chatbot")

st.write("""
Upload PDF, DOCX, TXT, CSV, XLSX, PPTX, JSON, XML,
Markdown or Images and ask questions using AI-powered RAG.
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
# MODEL LOADER
# =========================================================
@st.cache_resource
def load_model():

    try:
        pipe = pipeline(
            task="text2text-generation",
            model="google/flan-t5-base",
            tokenizer="google/flan-t5-base",
            max_new_tokens=200,
            do_sample=False
        )

    except Exception:

        pipe = pipeline(
            task="text-generation",
            model="google/flan-t5-base",
            max_new_tokens=200,
            do_sample=False
        )

    return pipe


# =========================================================
# TEXT EXTRACTION
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
                doc.page_content.strip()
                for doc in docs
                if doc.page_content.strip()
            ])

            os.unlink(tmp_path)

        # DOCX
        elif file_type == "docx":

            doc = Document(uploaded_file)

            text = "\n".join([
                para.text.strip()
                for para in doc.paragraphs
                if para.text.strip()
            ])

        # TXT / MD
        elif file_type in ["txt", "md"]:

            text = uploaded_file.read().decode(
                "utf-8",
                errors="ignore"
            )

        # JSON
        elif file_type == "json":

            uploaded_file.seek(0)

            data = json.load(uploaded_file)

            text = json.dumps(
                data,
                indent=2
            )

        # XML
        elif file_type == "xml":

            uploaded_file.seek(0)

            tree = ET.parse(uploaded_file)

            root = tree.getroot()

            xml_text = []

            for elem in root.iter():

                if elem.text:

                    cleaned = elem.text.strip()

                    if cleaned:
                        xml_text.append(cleaned)

            text = "\n".join(xml_text)

        # CSV
        elif file_type == "csv":

            uploaded_file.seek(0)

            df = pd.read_csv(uploaded_file)

            df = df.fillna("")

            rows = []

            for _, row in df.iterrows():

                row_text = " | ".join([
                    f"{col}: {str(row[col]).strip()}"
                    for col in df.columns
                    if str(row[col]).strip()
                ])

                rows.append(row_text)

            text = "\n".join(rows)

        # XLSX / XLS
        elif file_type in ["xlsx", "xls"]:

            uploaded_file.seek(0)

            excel_data = pd.read_excel(
                uploaded_file,
                sheet_name=None
            )

            all_rows = []

            for sheet_name, df in excel_data.items():

                df = df.fillna("")

                all_rows.append(
                    f"\n===== SHEET: {sheet_name} =====\n"
                )

                for _, row in df.iterrows():

                    row_text = " | ".join([
                        f"{col}: {str(row[col]).strip()}"
                        for col in df.columns
                        if str(row[col]).strip()
                    ])

                    if row_text.strip():
                        all_rows.append(row_text)

            text = "\n".join(all_rows)

        # PPTX
        elif file_type == "pptx":

            prs = Presentation(uploaded_file)

            slides_text = []

            for i, slide in enumerate(prs.slides):

                slides_text.append(
                    f"===== SLIDE {i+1} ====="
                )

                for shape in slide.shapes:

                    if hasattr(shape, "text"):

                        if shape.text.strip():

                            slides_text.append(
                                shape.text.strip()
                            )

            text = "\n".join(slides_text)

        # IMAGE OCR
        elif file_type in ["png", "jpg", "jpeg"]:

            image = Image.open(uploaded_file)

            reader = load_ocr()

            result = reader.readtext(image)

            text = "\n".join([
                item[1]
                for item in result
            ])

        return text.strip()

    except Exception as e:

        st.error(
            f"❌ Error reading file: {str(e)}"
        )

        return ""
# =========================================================
# VECTORSTORE
# =========================================================
@st.cache_resource(show_spinner=False)
def create_vectorstore(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_text(text)

    chunks = list(set([
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]))

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
# ASK QUESTION
# =========================================================
def ask_question(
    vectorstore,
    question,
    model
):

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
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
    if any(
        q in lower_question
        for q in summary_questions
    ):

        lines = []

        for line in context.split("\n"):

            line = line.strip()

            if (
                len(line) > 5
                and line not in lines
            ):
                lines.append(line)

        summary = "\n".join([
            f"• {line}"
            for line in lines[:20]
        ])

        return (
            f"### File Summary\n\n{summary}",
            docs
        )

    # =====================================================
    # PROMPT
    # =====================================================
    prompt = f"""
You are an intelligent AI document assistant.

Answer ONLY from the provided context.

Rules:
1. Give accurate answers.
2. Do not hallucinate.
3. If answer is not found say:
"I could not find that information in the uploaded file."
4. Keep answers short and clear.
5. Format answers cleanly.

Context:
{context}

Question:
{question}

Answer:
"""

    try:

        result = model(prompt)

        if isinstance(result, list):

            if "generated_text" in result[0]:

                answer = result[0][
                    "generated_text"
                ]

            elif "summary_text" in result[0]:

                answer = result[0][
                    "summary_text"
                ]

            else:

                answer = str(result[0])

        else:

            answer = str(result)

    except Exception as e:

        answer = (
            f"Error generating answer: "
            f"{str(e)}"
        )

    return answer, docs


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

    if (
        st.session_state.last_file
        != uploaded_file.name
    ):

        st.session_state.answer = ""
        st.session_state.docs = []

        st.session_state.last_file = (
            uploaded_file.name
        )

    with st.spinner(
        "📚 Reading file..."
    ):

        text = extract_text(
            uploaded_file
        )

    if text:

        with st.spinner(
            "🧠 Building AI knowledge base..."
        ):

            vectorstore = (
                create_vectorstore(text)
            )

            model = load_model()

        st.success(
            "✅ File uploaded successfully!"
        )

        file_type = (
            uploaded_file.name
            .split(".")[-1]
            .lower()
        )

        # =====================================
        # FILE PREVIEW
        # =====================================
        with st.expander(
            "📄 File Preview"
        ):

            if file_type == "csv":

                uploaded_file.seek(0)

                df = pd.read_csv(
                    uploaded_file
                )

                st.dataframe(df)

            elif file_type in [
                "xlsx",
                "xls"
            ]:

                uploaded_file.seek(0)

                excel_data = (
                    pd.read_excel(
                        uploaded_file,
                        sheet_name=None
                    )
                )

                for (
                    sheet_name,
                    df
                ) in excel_data.items():

                    st.subheader(
                        f"📑 Sheet: "
                        f"{sheet_name}"
                    )

                    st.dataframe(df)

            elif file_type in [
                "png",
                "jpg",
                "jpeg"
            ]:

                uploaded_file.seek(0)

                image = Image.open(
                    uploaded_file
                )

                st.image(
                    image,
                    use_container_width=True
                )

                st.write(
                    text[:3000]
                )

            elif file_type == "json":

                uploaded_file.seek(0)

                data = json.load(
                    uploaded_file
                )

                st.json(data)

            else:

                st.write(
                    text[:3000]
                )

        # =====================================
        # QUESTION INPUT
        # =====================================
        question = st.text_input(
            "💬 Ask a question"
        )

        if question:

            with st.spinner(
                "🔍 Finding answer..."
            ):

                (
                    answer,
                    docs
                ) = ask_question(
                    vectorstore,
                    question,
                    model
                )

                st.session_state.answer = (
                    answer
                )

                st.session_state.docs = (
                    docs
                )

        # =====================================
        # SHOW ANSWER
        # =====================================
        if st.session_state.answer:

            st.subheader(
                "🤖 Answer"
            )

            st.write(
                st.session_state.answer
            )

            with st.expander(
                "📌 Source Chunks Used"
            ):

                for (
                    i,
                    doc
                ) in enumerate(
                    st.session_state.docs
                ):

                    st.markdown(
                        f"### Chunk {i+1}"
                    )

                    st.write(
                        doc.page_content[
                            :1000
                        ]
                    )

                    st.divider()

    else:

        st.error(
            "❌ No readable "
            "content found."
        )

else:

    st.info(
        "📂 Upload a file "
        "to begin."
    )
