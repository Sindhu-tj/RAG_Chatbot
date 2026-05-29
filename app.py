import streamlit as st
import tempfile
import os
import pandas as pd

from PIL import Image
from docx import Document
from pptx import Presentation

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
# CSS
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

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================
st.title("🤖 Universal AI File Chatbot")

st.write(
    "Upload PDF, DOCX, TXT, CSV, XLSX, PPTX and ask questions using AI-powered RAG."
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

        # CSV
        elif file_type == "csv":

            df = pd.read_csv(uploaded_file)

            df = df.fillna("")

            text = df.to_string(index=False)

        # XLSX
        elif file_type == "xlsx":

            excel_data = pd.read_excel(
                uploaded_file,
                sheet_name=None,
                engine="openpyxl"
            )

            all_text = []

            for sheet_name, df in excel_data.items():

                df = df.fillna("")

                all_text.append(
                    f"\n\n===== SHEET: {sheet_name} =====\n"
                )

                all_text.append(
                    df.to_string(index=False)
                )

            text = "\n".join(all_text)

        # PPTX
        elif file_type == "pptx":

            prs = Presentation(uploaded_file)

            slide_text = []

            for slide in prs.slides:

                for shape in slide.shapes:

                    if hasattr(shape, "text"):

                        if shape.text.strip():

                            slide_text.append(shape.text)

            text = "\n".join(slide_text)

        else:

            st.error("Unsupported file type")

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
        chunk_size=800,
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
# MODEL
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
# QUESTION ANSWERING
# =========================================================
def ask_question(vectorstore, question, model):

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    docs = retriever.invoke(question)

    context = "\n\n".join([
        doc.page_content
        for doc in docs
    ])

    # SMART SUMMARY
    if question.lower() in [
        "what is in the file",
        "summarize file",
        "summary",
        "about file"
    ]:

        return (
            context[:1000],
            docs
        )

    prompt = f"""
Answer ONLY from the context.

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

    result = model(prompt)[0]["generated_text"]

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
        "pptx"
    ]
)

# =========================================================
# MAIN
# =========================================================
if uploaded_file:

    with st.spinner("📚 Reading file..."):

        text = extract_text(uploaded_file)

    if text:

        with st.spinner("🧠 Building AI knowledge base..."):

            vectorstore = create_vectorstore(text)

            model = load_model()

        st.success("✅ File uploaded successfully!")

        # Preview
        with st.expander("📄 File Preview"):

            st.write(text[:3000])

        # Question
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

            st.subheader("🤖 Answer")

            st.write(answer)

            # Sources
            with st.expander("📌 Source Chunks Used"):

                for i, doc in enumerate(docs):

                    st.markdown(f"### Chunk {i+1}")

                    st.write(doc.page_content[:700])

                    st.divider()

    else:

        st.error("❌ No readable content found.")

else:

    st.info("📂 Upload a file to begin.")