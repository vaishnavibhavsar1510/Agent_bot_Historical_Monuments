# app.py  ──────────────────────────────────────────────────────────────
import os, re, streamlit as st
from dotenv import load_dotenv

# LangChain/LangGraph imports
from langchain_core.messages import HumanMessage, AIMessage
from backend.app.langgraph_workflow import compiled_chat_graph, ChatState
from backend.app.otp import find_email # Only needed for initial greeting condition

load_dotenv() # local .env for dev
# Adding a comment to force refresh.

# ── Inject the key for LangChain/OpenAI helpers ───────────────────────
os.environ["OPENAI_API_KEY"] = (
    os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
)

# ── Page config & CSS (use your existing big CSS block) ───────────────
st.set_page_config(page_title="Historical Monument Agent",
                   page_icon="🏛️", layout="centered")
st.markdown("""<style> … YOUR  CSS  BLOCK … </style>""",
            unsafe_allow_html=True)

# ── Session-state defaults ────────────────────────────────────────────
for k, v in {
    "messages": [],
    "awaiting_otp": False,
    "otp_attempts": 0,
    "email": None,
    "last_monument_query": None,
    "user_input": None,
    "awaiting_user_choice": False, # NEW: for yes/no responses
}.items():
    st.session_state.setdefault(k, v)

# ── Chat-bubble helper ────────────────────────────────────────────────
def bubble(role: str, text: str):
    avatar = "🤖" if role == "assistant" else None
    with st.chat_message(role, avatar=avatar):
        st.markdown(text)

# ── Main loop ─────────────────────────────────────────────────────────
def main() -> None:
    st.title("🏛️ Historical Monument Agent")

    # 1) greet on first load
    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hey there 👋  Ask me about any historical monument!"
        })

    # 2) replay history
    for m in st.session_state.messages:
        bubble(m["role"], m["content"])

    # 3) free-text input (hidden while OTP form showing)
    fresh_text = None
    if not st.session_state.awaiting_otp:
        prompt = "Message…"
        fresh_text = st.chat_input(prompt)

    if fresh_text:
        st.session_state.user_input = fresh_text
        # Add user message to history. This will be consumed by LangGraph.
        st.session_state.messages.append({"role": "user", "content": fresh_text})
        bubble("user", fresh_text)

    # 4) OTP form
    if st.session_state.awaiting_otp:
        with st.form("otp_form"):
            otp_code = st.text_input("Enter the 6-digit code:")
            c1, c2 = st.columns(2)
            verify = c1.form_submit_button("Verify OTP")
            cancel = c2.form_submit_button("Cancel")

            if cancel:
                # Reset state completely for cancellation
                st.session_state.update(
                    awaiting_otp=False,
                    otp_attempts=0,
                    email=None,
                    last_monument_query=None,
                    user_input=None,
                    awaiting_user_choice=False,
                    messages=[{ # Reset messages to initial greeting
                        "role": "assistant",
                        "content": "Email verification cancelled. How else can I help?"
                    }]
                )
                st.rerun()
            if verify and otp_code:
                st.session_state.user_input = otp_code # This will be processed by LangGraph

    # 5) Process user input through LangGraph
    # Only run LangGraph if there's new user input or if we are in an OTP flow
    if st.session_state.user_input is not None or st.session_state.awaiting_otp:
        # Convert st.session_state.messages to LangChain format for ChatState
        langchain_messages = []
        for msg_dict in st.session_state.messages:
            if msg_dict["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg_dict["content"]))
            elif msg_dict["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg_dict["content"]))

        current_chat_state = ChatState(
            messages=langchain_messages,
            user_input=st.session_state.user_input,
            awaiting_otp=st.session_state.awaiting_otp,
            email=st.session_state.email,
            otp_attempts=st.session_state.otp_attempts,
            last_monument_query=st.session_state.last_monument_query,
            initial_response_sent=(len(st.session_state.messages) > 0), # True if messages exist
            awaiting_user_choice=st.session_state.awaiting_user_choice
        )

        # Invoke the LangGraph
        with st.spinner("Thinking..."):
            result_state_dict = compiled_chat_graph.invoke(current_chat_state.model_dump())
            result_state = ChatState.model_validate(result_state_dict)

        # Update Streamlit session state from LangGraph result
        st.session_state.user_input = None # Consume after processing
        st.session_state.awaiting_otp = result_state.awaiting_otp
        st.session_state.email = result_state.email
        st.session_state.otp_attempts = result_state.otp_attempts
        st.session_state.last_monument_query = result_state.last_monument_query
        st.session_state.awaiting_user_choice = result_state.awaiting_user_choice

        # Add bot's response to messages if available and not already added
        if result_state.response and not any(m["content"] == result_state.response and m["role"] == "assistant" for m in st.session_state.messages):
            st.session_state.messages.append({"role": "assistant", "content": result_state.response})

        # Re-run the app to update UI based on new state
        st.rerun()

# ── run ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
