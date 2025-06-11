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
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# New imports for conversational memory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

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
def _build_qa_chain() -> ConversationalRetrievalChain:
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

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, api_key=key)

    # 1. Contextualize question
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is. "
        "If the user is asking for 'more places' or similar, assume they are asking for more monuments in the last mentioned location."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 2. Answer question
    qa_system_prompt = (
        "You are an AI assistant specialized in historical monuments. Use "
        "the following pieces of retrieved context to answer the "
        "question. If you don't know the answer, just say that you "
        "don't know. Use three sentences maximum and keep the answer "
        "concise. If the question refers to 'them' or 'it', infer the specific monument(s) from the chat history and answer strictly about those. Do NOT mention monuments or locations not explicitly discussed or relevant to the immediate context. If the question is about visiting multiple places, provide a realistic assessment based on travel times and locations.\n\nContext: {context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"), # This is important to pass the history to the final QA chain
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

# ── Public helper for Streamlit UI ──────────────────────────────────────────
def answer_monument_query(query: str, chat_history: list) -> str | None:
    """
    Return a concise answer for *query* using the monument knowledge base.
    Returns None if no specific information is found.
    """
    qa_chain = _build_qa_chain()

    # The new chain expects a dictionary with 'input' and 'chat_history'
    result = qa_chain.invoke({"input": query, "chat_history": chat_history})

    response_text = ""
    # The output structure of create_retrieval_chain is {'input': ..., 'chat_history': ..., 'answer': ..., 'context': ...}
    if isinstance(result, dict) and "answer" in result:
        response_text = result["answer"]
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
