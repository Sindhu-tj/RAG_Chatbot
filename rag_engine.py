from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from transformers import pipeline

import tempfile
import os


# ==========================================
# LOAD PDF + CREATE VECTOR DATABASE
# ==========================================
def load_and_index(pdf_file):

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name

    # Load PDF
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    # Delete temp file
    os.unlink(tmp_path)

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    # Embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create FAISS vector database
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore


# ==========================================
# BUILD RAG QA CHAIN
# ==========================================
def build_qa_chain(vectorstore):

    # IMPORTANT FIX:
    # Use text2text-generation pipeline properly
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256,
        temperature=0.3
    )

    # Wrap with LangChain
    llm = HuggingFacePipeline(
        pipeline=pipe
    )

    # Retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    # Prompt template
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

    # Convert retrieved docs into text
    def format_docs(docs):
        return "\n\n".join(
            doc.page_content for doc in docs
        )

    # Build RAG chain
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

    # Generate answer
    answer = chain.invoke(question)

    # Retrieve source chunks
    source_docs = retriever.invoke(question)

    return answer, source_docs