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

        # PDF
        if file_type == "pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                path = tmp.name
            loader = PyPDFLoader(path)
            docs = loader.load()
            text = "\n".join([d.page_content for d in docs])
            os.unlink(path)

        # DOCX
        elif file_type == "docx":
            doc = Document(uploaded_file)
            text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

        # TXT
        elif file_type == "txt":
            text = uploaded_file.read().decode("utf-8", errors="ignore")

        # JSON / XML / MD
        elif file_type in ["json", "xml", "md"]:
            text = uploaded_file.read().decode("utf-8", errors="ignore")

        # CSV
        elif file_type == "csv":
            df = pd.read_csv(uploaded_file).fillna("")
            st.session_state.df = df
            text = df.to_string(index=False)

        # EXCEL
        elif file_type in ["xlsx", "xls"]:
            df = pd.read_excel(uploaded_file).fillna("")
            st.session_state.df = df
            text = df.to_string(index=False)

        # PPTX
        elif file_type == "pptx":
            prs = Presentation(uploaded_file)
            slides = []
            for i, slide in enumerate(prs.slides):
                slides.append(f"\nSLIDE {i+1}")
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        slides.append(shape.text)
            text = "\n".join(slides)

        # IMAGE OCR
        elif file_type in ["png", "jpg", "jpeg"]:
            image = Image.open(uploaded_file)
            reader = load_ocr()
            results = reader.readtext(image)
            text = "\n".join([r[1] for r in results])
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
    docs = [
        LangDocument(page_content=c, metadata={"chunk_id": i})
        for i, c in enumerate(chunks)
    ]
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


# =========================================================
# HELPER: format a list of values cleanly
# =========================================================
def format_list(values):
    """Return a sorted, deduplicated, newline-separated list string."""
    clean = sorted(set(str(v).strip() for v in values if str(v).strip()))
    return "\n".join(f"• {item}" for item in clean)


# =========================================================
# SMART ANSWER ENGINE  (fully rewritten)
# =========================================================
def smart_answer(file_type, question, text, vectorstore, model):

    q = question.lower().strip()
    df = st.session_state.df

    # =========================================================
    # CSV / EXCEL  — structured data path
    # =========================================================
    if file_type in ["csv", "xlsx", "xls"] and df is not None:

        col_names_lower = {c.lower(): c for c in df.columns}

        # ── LIST / SHOW ALL VALUES IN A COLUMN ──────────────
        # e.g. "list all company names", "show me all companies"
        list_triggers = ["list", "show", "give", "all", "display", "what are", "names"]
        is_list_request = any(t in q for t in list_triggers)

        if is_list_request:
            # Try to find which column the user is asking about
            matched_col = None
            for col_lower, col_real in col_names_lower.items():
                # check if any word in the column name appears in the question
                col_words = col_lower.replace("_", " ").split()
                if any(w in q for w in col_words):
                    matched_col = col_real
                    break

            if matched_col:
                values = df[matched_col].dropna().tolist()
                formatted = format_list(values)
                return {
                    "type": "text",
                    "content": f"**{matched_col}** ({len(set(values))} unique values):\n\n{formatted}"
                }

            # If no specific column matched but user just says "list all" → show summary
            # fall through to summary below

        # ── SUMMARY / OVERVIEW ──────────────────────────────
        summary_triggers = ["summary", "overview", "describe", "about this file",
                            "what is this", "what does this file", "tell me about"]
        if any(t in q for t in summary_triggers) or (
            is_list_request and not any(w in q for w in col_names_lower)
        ):
            return {
                "type": "table_summary",
                "rows": df.shape[0],
                "cols": df.shape[1],
                "columns": list(df.columns),
                "preview": df.head(20)
            }

        # ── COLUMN NAMES ────────────────────────────────────
        if "column" in q and ("name" in q or "what" in q or "list" in q):
            return {
                "type": "text",
                "content": f"**Columns ({len(df.columns)}):**\n\n" +
                           "\n".join(f"• {c}" for c in df.columns)
            }

        # ── ROW / ENTRY COUNT ────────────────────────────────
        if any(w in q for w in ["how many rows", "row count", "how many entries",
                                  "how many records", "total rows", "total entries"]):
            return {
                "type": "text",
                "content": f"The file contains **{df.shape[0]} rows** and **{df.shape[1]} columns**."
            }

        # ── COUNT UNIQUE VALUES IN A COLUMN ─────────────────
        if "how many" in q or "count" in q:
            for col_lower, col_real in col_names_lower.items():
                col_words = col_lower.replace("_", " ").split()
                if any(w in q for w in col_words):
                    n = df[col_real].nunique()
                    return {
                        "type": "text",
                        "content": f"There are **{n} unique values** in **{col_real}**."
                    }
            return {
                "type": "text",
                "content": f"The file contains **{df.shape[0]} rows** and **{df.shape[1]} columns**."
            }

        # ── SEARCH / FILTER ──────────────────────────────────
        # Extract search terms (words not in stop words)
        stop = {"find", "search", "show", "get", "give", "where", "is", "are",
                "the", "a", "an", "me", "all", "any", "about", "for", "with",
                "in", "of", "and", "or", "that", "which", "what", "list"}
        search_terms = [w for w in q.split() if w not in stop and len(w) > 2]

        if search_terms:
            matched = pd.DataFrame()
            for col in df.columns:
                for term in search_terms:
                    temp = df[
                        df[col].astype(str).str.lower().str.contains(term, na=False)
                    ]
                    matched = pd.concat([matched, temp])
            matched = matched.drop_duplicates()

            if len(matched) > 0:
                return {"type": "table", "data": matched.head(50)}

        return {
            "type": "text",
            "content": "No matching data found. Try rephrasing or use column names from the file."
        }

    # =========================================================
    # JSON
    # =========================================================
    elif file_type == "json":
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                keys_preview = list(data.keys())[:20]
                return {
                    "type": "text",
                    "content": f"**JSON keys ({len(data)}):**\n\n" +
                               "\n".join(f"• {k}" for k in keys_preview)
                }
            elif isinstance(data, list):
                return {
                    "type": "text",
                    "content": f"JSON is a list with **{len(data)} items**.\n\n"
                               f"First item preview:\n```\n{json.dumps(data[0], indent=2)[:1000]}\n```"
                }
        except Exception:
            return {"type": "text", "content": "Could not parse JSON structure."}

    # =========================================================
    # IMAGE OCR
    # =========================================================
    elif file_type in ["png", "jpg", "jpeg"]:
        return {
            "type": "text",
            "content": f"**Extracted text from image:**\n\n{text[:3000]}"
        }

    # =========================================================
    # PPTX
    # =========================================================
    elif file_type == "pptx":
        prompt = (
            "Summarize this PowerPoint presentation slide by slide.\n\n"
            f"TEXT:\n{text[:4000]}"
        )
        result = model(prompt)[0]["generated_text"]
        return {"type": "text", "content": result}

    # =========================================================
    # GENERAL RAG (PDF, DOCX, TXT, MD, XML)
    # =========================================================
    else:
        docs = vectorstore.similarity_search(question, k=4)
        context = "\n\n".join([d.page_content for d in docs])

        prompt = (
            "You are a precise document assistant. "
            "Answer ONLY from the context below. "
            "If the answer is not in the context, say 'Information not found in document'.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            "ANSWER:"
        )

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
        "pdf", "docx", "txt", "csv", "xlsx", "xls",
        "pptx", "json", "xml", "md", "png", "jpg", "jpeg"
    ]
)

# =========================================================
# FILE LOADED
# =========================================================
if uploaded_file:

    file_type = uploaded_file.name.split(".")[-1].lower()

    # RESET on new file
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
                st.write(f"**Column Names:** {', '.join(df.columns)}")

        # =====================================================
        # PREVIEW
        # =====================================================
        st.subheader("📄 File Preview")

        if file_type in ["csv", "xlsx", "xls"]:
            st.dataframe(
                st.session_state.df.head(20),
                use_container_width=True,
                height=400
            )
        else:
            st.text(text[:3000])

        # =====================================================
        # QUESTION INPUT
        # =====================================================
        question = st.text_input("💬 Ask a question about the file")

        # =====================================================
        # AI ANSWER
        # =====================================================
        if question:

            with st.spinner("Thinking..."):
                response = smart_answer(
                    file_type, question, text, vectorstore, model
                )

            st.subheader("🤖 AI Answer")

            # TABLE SUMMARY
            if response["type"] == "table_summary":
                st.success("File analyzed successfully.")
                st.markdown(
                    f"- **Rows:** {response['rows']}\n"
                    f"- **Columns:** {response['cols']}\n"
                    f"- **Column Names:** {', '.join(response['columns'])}"
                )
                st.dataframe(
                    response["preview"],
                    use_container_width=True,
                    height=400
                )

            # FILTERED TABLE
            elif response["type"] == "table":
                st.info(f"Found **{len(response['data'])}** matching records.")
                st.dataframe(
                    response["data"],
                    use_container_width=True,
                    height=400
                )

            # PLAIN TEXT (includes lists)
            elif response["type"] == "text":
                st.markdown(response["content"])

            # RAG ANSWER
            elif response["type"] == "rag":
                st.write(response["content"])
                with st.expander("📌 Source Chunks Used"):
                    st.write(f"{len(response['docs'])} relevant chunks retrieved.")
                    for i, d in enumerate(response["docs"]):
                        st.markdown(f"### Chunk {i+1}")
                        st.write(d.page_content[:800])

    else:
        st.error("No content could be extracted from the file.")

# =========================================================
# NO FILE
# =========================================================
else:
    st.info("⬆️ Upload a file to get started.")