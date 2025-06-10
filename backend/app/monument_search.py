# backend/app/monument_search.py
"""
Light-weight Retrieval-QA wrapper over a local JSON list of monuments.

• Loads monument data once  (st.cache_data)
• Builds an in-memory FAISS index once (st.cache_resource)
• Exposes answer_monument_query() for the Streamlit app
"""

from __future__ import annotations
import json, os, re
from pathlib import Path
import streamlit as st

from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# ── Locate data/monuments.json ──────────────────────────────────────────────
ROOT_DIR  = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "monuments.json"

# ── Helper: fetch key from env or st.secrets ────────────────────────────────
def _openai_key() -> str:
    # prefer env-var so REPL/tests work; fall back to st.secrets
    return (
        os.getenv("OPENAI_API_KEY")
        or st.secrets.get("OPENAI_API_KEY", "")
    )

# ── Cache monument JSON ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="📚 Loading monument data…")
def _load_monuments() -> list[dict]:
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)

# ── Cache FAISS index + RetrievalQA chain ───────────────────────────────────
@st.cache_resource(show_spinner="🔧 Building FAISS index…")
def _build_qa_chain() -> RetrievalQA:
    monuments  = _load_monuments()
    texts      = [m["description"] for m in monuments]
    metadatas  = [{"name": m["name"], "location": m["location"]} for m in monuments]
    key        = _openai_key()

    vs = FAISS.from_texts(
        texts,
        embedding=OpenAIEmbeddings(openai_api_key=key),
        metadatas=metadatas,
    )
    retriever = vs.as_retriever()

    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, api_key=key),
        retriever=retriever,
    )

# ── Public helper for Streamlit UI ──────────────────────────────────────────
def answer_monument_query(query: str) -> str | None:
    """
    Return a concise answer for *query* using the monument knowledge base.
    Returns None if no specific information is found.
    """
    qa_chain = _build_qa_chain()

    try:
        result = qa_chain.invoke({"query": query})
    except TypeError:
        result = qa_chain.invoke(query)

    response_text = ""
    if isinstance(result, dict):
        for k in ("result", "output_text", "answer"):
            if isinstance(result.get(k), str):
                response_text = result[k]
                break
        if not response_text:
            response_text = str(result)
    else:
        response_text = str(result)

    # Check for common "no information" phrases that indicate a lack of specific match
    no_info_patterns = [
        r"I don't have information",
        r"I cannot find information",
        r"I'm sorry, but I don't have details",
        r"I am sorry, but I cannot provide information",
        r"no specific information about",
        r"I cannot find any information about",
        r"I am unable to find any information about",
        r"I don't have details on",
        r"I couldn't find any information",
        r"I am not able to find details on"
    ]
    if any(re.search(pattern, response_text, re.IGNORECASE) for pattern in no_info_patterns):
        return None  # Explicitly return None if no specific information is found

    return response_text
