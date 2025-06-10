# backend/app/chat.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.app.langgraph_workflow import compiled_chat_graph, ChatState
from langchain_core.messages import HumanMessage, AIMessage
import logging

# ------------------- Logging Setup -------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ------------------- FastAPI Router -------------------
router = APIRouter()

class ChatRequest(BaseModel):
    session_id: str        # A unique identifier for each user/session
    user_query: str        # Whatever the user just typed (monument question, email, or OTP)

class ChatResponse(BaseModel):
    message: str           # The single bot reply to return

# In‐memory store of ChatState objects, keyed by session_id.
# In production you could store this in Redis or a database instead.
SESSION_STATES: dict[str, ChatState] = {}

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Single endpoint that handles:
      • A "normal" monument question
      • An email address (to trigger OTP)
      • A 6-digit OTP code (to verify & send final email)
    The ChatState for each session_id is persisted in SESSION_STATES.
    """
    session_id = request.session_id.strip()
    user_input = request.user_query.strip()

    # 1) Load or initialize the ChatState for this session_id
    if session_id in SESSION_STATES:
        state: ChatState = SESSION_STATES[session_id]
    else:
        # First time we see this session_id → create a brand‐new ChatState
        # Start with an initial greeting in messages (optional; you can omit or customize):
        state = ChatState(
            user_input="",
            messages=[HumanMessage(content="Hey, I am a historical agent AI. You can ask anything about historical monuments!")]
        )

    # 2) Append the user's latest message to state.messages,
    #    and store it in state.user_input so our LangGraph graph can see it.
    state.user_input = user_input
    state.messages.append(HumanMessage(content=user_input))

    try:
        # 3) Invoke the compiled LangGraph workflow (async in this case).
        #    We pass the state as a dict.  The graph will internally check:
        #      • If awaiting_email=True and `user_input` is a valid email → go to send_otp node
        #      • If awaiting_otp=True and `user_input` is a 6‐digit code → go to verify_otp node
        #      • Otherwise → treat as monument query (run RAG and set awaiting_email=True)
        result_dict = await compiled_chat_graph.ainvoke(state.model_dump())

        # 4) Convert the returned dict back into a ChatState object
        new_state = ChatState.model_validate(result_dict)

        # 5) Persist the updated ChatState for this session_id
        SESSION_STATES[session_id] = new_state

        # 6) Extract "the latest bot reply" from new_state.  We look for:
        #      • The last AIMessage in new_state.messages (most common)
        #      • Otherwise, fallback to new_state.response if it exists
        bot_reply = None

        if new_state.messages:
            last_msg = new_state.messages[-1]
            if isinstance(last_msg, AIMessage):
                bot_reply = last_msg.content

        # If we didn't find an AIMessage, check if there's a 'response' field:
        if (bot_reply is None) and hasattr(new_state, "response"):
            # Pydantic BaseModel: new_state.dict().get("response")
            resp = new_state.model_dump().get("response")
            if isinstance(resp, str) and resp:
                bot_reply = resp

        # If still no reply, return a generic fallback
        if bot_reply is None:
            logger.warning("LangGraph finished without an AIMessage or 'response' field.")
            bot_reply = "Sorry, I couldn't generate a response."

        return ChatResponse(message=bot_reply)

    except Exception as e:
        # Log full traceback for debugging
        logger.error("Error running LangGraph workflow", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
