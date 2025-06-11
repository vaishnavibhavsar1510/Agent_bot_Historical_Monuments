# This is a test comment to force a file refresh.
"""
LangGraph state-machine for the historical-monument chatbot
with e-mail + OTP verification flow.
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict
import re

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from backend.app.config import settings
from backend.app.monument_search import answer_monument_query
from backend.app.otp import (
    generate_otp,
    store_otp,
    retrieve_stored_otp,
    delete_otp,
    is_valid_email,     # quick syntactic check
    find_email,         # ← NEW helper: pull e-mail out of a sentence
    extract_otp         # ← NEW helper: pull 6-digit code out of text
)
from backend.app.email_utils import (
    send_otp_email,     # keep OTP template
    send_plain_email    # ← NEW helper: generic, no OTP footer
)

# --------------------------------------------------------------------------- #
# Logging & LLM
# --------------------------------------------------------------------------- #

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=settings.openai_api_key)

# --------------------------------------------------------------------------- #
# Chat-state dataclass
# --------------------------------------------------------------------------- #

class ChatState(BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)
    user_input: Optional[str] = None

    awaiting_otp: bool = False
    email: Optional[str] = None
    otp_attempts: int = 0

    monument_results: List[Dict] = Field(default_factory=list)
    response: Optional[str] = None
    next_step: str = "process_user_input"
    last_monument_query: Optional[str] = None
    initial_response_sent: bool = False
    awaiting_user_choice: bool = False

# --------------------------------------------------------------------------- #
#  Node: process_user_input
# --------------------------------------------------------------------------- #

def process_user_input(state: ChatState) -> ChatState:
    """
    Process user input and handle routing based on the current state.
    - If it's the very first user input, send an initial greeting.
    - If user provides an email at any point, handle it
    - If awaiting OTP, process OTP input
    - Otherwise, treat as a monument query.
    """
    logger.info(
        "Processing user_input; awaiting_otp=%s initial_response_sent=%s input=%r",
        state.awaiting_otp, state.initial_response_sent, state.user_input,
    )

    # Record message if present
    if state.user_input:
        state.messages.append(HumanMessage(content=state.user_input))

    # Handle user choice if awaiting it
    if state.awaiting_user_choice:
        state.next_step = "handle_user_choice"
        return state

    # If it's the very first user input and no email is detected, send a generic greeting
    if not state.initial_response_sent and not find_email(state.user_input):
        state.next_step = "initial_greeting_node"
        return state

    # Handle OTP verification if we're waiting for it
    if state.awaiting_otp:
        if not state.user_input:
            state.next_step = END
            return state

        code = extract_otp(state.user_input, digits=6)
        if code:
            logger.info("Extracted OTP %s → process_otp_input", code)
            state.next_step = "process_otp_input"
            return state

        # Still no 6-digit number
        msg = "Please enter the 6-digit code that was e-mailed to you."
        state.messages.append(AIMessage(content=msg))
        state.response = msg
        state.next_step = END
        return state

    # Check for email in the input at any point (after initial greeting)
    # This logic should only run if the initial greeting has already been sent
    if state.initial_response_sent and state.user_input:
        email = find_email(state.user_input)
        if email:
            state.email = email
            state.next_step = "send_otp"
            logger.info("Email extracted → send_otp")
            state.user_input = None  # Consume the email input
            return state

    # If no email found and not awaiting OTP, treat as monument query
    state.next_step = "check_query_type"
    state.user_input = None
    return state

# --------------------------------------------------------------------------- #
#  Remaining nodes (monument flow, OTP flow, etc.)
# --------------------------------------------------------------------------- #

def generate_monument_response(state: ChatState) -> ChatState:
    """
    Generates a response for a monument-related query.
    """
    query = state.messages[-1].content # Always use the current user input as the query for answer_monument_query
    logger.info(f"Generating monument response for query: {query}")
    if query:
        # Ensure answer_monument_query uses the current last_monument_query (which might be rephrased)
        answer = answer_monument_query(query, state.messages)
        if answer:
            reply = (
                answer
                + " If you'd like more details e-mailed to you, please feel free to provide your email address in the chat."
            )
            state.messages.append(AIMessage(content=reply))
            state.response = reply
        else:
            # If answer is empty, it means no specific information was found for the query
            reply = (
                "I couldn't find specific information for that location in my current knowledge base. "
                "However, I can still assist you with general information about other well-known historical monuments "
                "like the Taj Mahal, Eiffel Tower, or the Great Wall of China. "
                "If you'd like more details e-mailed to you for these or other known monuments, "
                "please feel free to provide your email address in the chat."
            )
            state.messages.append(AIMessage(content=reply))
            state.response = reply
    else:
        reply = "I'm sorry, I don't have enough information to answer that monument query."
        state.messages.append(AIMessage(content=reply))
        state.response = reply
    state.next_step = "check_query_type"
    return state


def check_query_type(state: ChatState) -> ChatState:
    query = state.messages[-1].content if state.messages else ""
    last_monument_query = state.last_monument_query

    recent_chat_history = ""
    # Get last 4 messages (2 human, 2 AI) for context, or fewer if not available
    for msg in state.messages[-4:]:
        if isinstance(msg, HumanMessage):
            recent_chat_history += f"Human: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            recent_chat_history += f"AI: {msg.content}\n"

    # Emphasize the historical monument context and how to handle follow-ups
    prompt_instruction = """You are an AI assistant specialized in historical monuments. Your task is to classify user queries into 'MONUMENT' or 'GENERAL'.

    **CRITICAL**: You MUST use the *Recent Chat History* to understand the context. If the *Current user question* is a follow-up related to any historical monument or its associated travel/logistical details that have been discussed in the *Recent Chat History* (even if it uses pronouns like 'them' or 'it'), or if the query asks about historical monuments or places to visit in a specific country (e.g., "monuments in Japan", "places to visit in Italy"), you MUST classify it as 'MONUMENT'.

    Only classify as 'GENERAL' if the *Current user question* is unequivocally unrelated to historical monuments or their associated travel/logistics, or if there is no prior historical monument context in the conversation.
    """

    query_examples = """Examples of MONUMENT queries:
  - 'Tell me about the Eiffel Tower' (direct monument query)
  - 'Can I visit them within a week time' (if 'them' refers to previously discussed monuments)
  - 'What's the history of Taj Mahal' (direct monument query)
  - 'How far is it from Paris?' (if a monument in Paris was previously discussed)
  - 'When are they open?' (contextual follow-up about monument operating hours)
  - 'How much does it cost to enter?' (contextual follow-up about monument entry fees)
  - 'Are they wheelchair accessible?' (contextual follow-up about monument accessibility)
  - 'What are the best times to visit them?' (contextual follow-up about monument visiting times)
  - 'How long does it take to visit them?' (contextual follow-up about monument visit duration)
  - 'monuments in Japan' (query about monuments in a specific country)
  - 'places to visit in Italy' (query about places to visit in a specific country)

Examples of GENERAL queries:
  - 'How to improve a tennis game?' (irrelevant)
  - 'What's the weather today?' (irrelevant)
  - 'Tell me a joke' (irrelevant)
"""

    full_prompt = (
        f"{prompt_instruction}\n\n"
        f"Recent Chat History:\n{recent_chat_history}\n"
        f"Current user question: {query}\n\n"
        f"{query_examples}\n\n"
        "Your response must be only 'MONUMENT' or 'GENERAL'."
    )

    classification = llm.invoke(full_prompt).content.strip().upper()

    if "MONUMENT" in classification:
        query_type = "MONUMENT"
        logger.info(f"LLM classified query '{query}' as MONUMENT.")
        # Update last_monument_query with the current query if it's a monument query
        state.last_monument_query = query
        state.next_step = "generate_monument_response"
    else:
        query_type = "GENERAL"
        logger.info(f"LLM classified query '{query}' as GENERAL.")
        state.next_step = "generate_non_monument_response"
    return state


def generate_non_monument_response(state: ChatState) -> ChatState:
    question = state.messages[-1].content
    prompt = (
        f"The user asked: {question}\n"
        "You are a helpful chatbot specializing only in historical monuments. "
        "Your goal is to politely inform the user that you can only answer questions about historical monuments "
        "and ask them to rephrase their question to be about a historical monument. "
        "DO NOT attempt to answer questions that are not about historical monuments. "
        "Your response should be concise and clearly redirect the user."
    )
    answer = llm.invoke(prompt).content
    reply = answer
    state.messages.append(AIMessage(content=reply))
    state.response = reply
    state.next_step = END
    return state


def send_otp_step(state: ChatState) -> ChatState:
    email = state.email
    otp = generate_otp()
    store_otp(email, otp, ttl_seconds=300)

    logger.info("Generated OTP: %s for email: %s", otp, email)

    if send_otp_email(email, otp):
        msg = (
            f"Thank you. An OTP has been sent to {email}. "
            "Please enter the 6-digit code here to verify your email "
            "and receive more details."
        )
        state.awaiting_otp = True
    else:
        msg = (
            f"Sorry, I couldn't send the OTP to {email}. "
            "Please check the address or provide a different one."
        )
        state.next_step = END

    state.response = msg
    state.messages.append(AIMessage(content=msg))
    return state


def process_otp_input(state: ChatState) -> ChatState:
    # Use state.user_input to extract the OTP, as it comes directly from the form submission
    code = extract_otp(state.user_input, digits=6) or ""
    email = state.email
    stored = retrieve_stored_otp(email)

    logger.info("User entered OTP: %s, Stored OTP: %s for email: %s", code, stored, email)

    if stored and code == stored:
        delete_otp(email)
        state.awaiting_otp = False
        state.next_step = "final_confirmation"
        msg = "Thank you! Your email is verified. I will send more details shortly."
        state.response = msg
        state.messages.append(AIMessage(content=msg))
        return state

    # Incorrect / expired
    state.otp_attempts += 1
    if state.otp_attempts >= 3 or stored is None:
        msg = "Too many incorrect attempts or code expired. Email verification failed."
        state.awaiting_otp = False
        state.next_step = "end_conversation"
    else:
        msg = "That code is incorrect. Please try again."
        state.next_step = END

    state.response = msg
    state.messages.append(AIMessage(content=msg))
    return state


def final_confirmation(state: ChatState) -> ChatState:
    """
    Compose a detailed guide and e-mail it *without* OTP footer.
    """
    email = state.email or ""
    monument_query = state.last_monument_query
    monument_answer = None # Initialize monument_answer to None

    email_subject = "Details from your Historical Monument Agent"
    email_body = ""

    logger.info("Final confirmation initiated for email: %s, last_monument_query: %r", email, monument_query)
    logger.info("Type of monument_query: %s", type(monument_query))

    if monument_query:
        # Attempt to get the monument answer
        try:
            monument_answer = answer_monument_query(monument_query, state.messages)
            logger.info("Found monument answer for email: %s", monument_answer)
        except Exception as e:
            logger.error("Error searching for monument details for email: %s", e)
            monument_answer = None # Ensure it's None on error
            email_body = (
                "Thank you for verifying your email. Unfortunately, I encountered an issue "
                "retrieving details for your last monument query. "
                "Please feel free to ask me about other historical monuments in the chat."
            )
    
    if monument_answer:
        email_subject = f"Details about {monument_query.title()}"
        email_body = (
            f"Dear user,\n\nHere are the details you requested for '{monument_query.title()}':\n\n"
            f"{monument_answer}\n\n"
            "If you have any more questions, feel free to ask!"
        )
    else:
        # If no specific monument query or search failed, and email_body wasn't set by an error
        if not email_body:
            email_body = (
                "Thank you for verifying your email. I can provide details about historical monuments. "
                "Please ask me about a specific monument in the chat, and I can email you more information."
            )

    if send_plain_email(email, email_subject, email_body):
        msg = "Thank you! Your email is verified. The details have been sent to your email."
        state.response = msg
        state.messages.append(AIMessage(content=msg))
        state.next_step = END
    else:
        msg = "Email verification successful, but I failed to send the detailed email. Please try again later."
        state.response = msg
        state.messages.append(AIMessage(content=msg))
        state.next_step = END # Or potentially a retry mechanism

    return state


def end_conversation(state: ChatState) -> ChatState:
    state.next_step = END
    return state


def initial_greeting_node(state: ChatState) -> ChatState:
    """
    Generates the initial greeting message and sets initial_response_sent to True.
    """
    
    msg = "Hello! How can I assist you today?"
    state.messages.append(AIMessage(content=msg))
    state.response = msg
    state.initial_response_sent = True
    state.next_step = END
    return state


def handle_user_choice(state: ChatState) -> ChatState:
    user_input = state.messages[-1].content.strip().lower()
    state.awaiting_user_choice = False # Reset the flag after processing

    if user_input == "yes":
        msg = "Great! How can I help you further? Feel free to ask any monument or general travel questions."
        state.next_step = END
    elif user_input == "no":
        msg = "No problem. Feel free to ask if you change your mind, or if you have any other questions."
        state.next_step = END
    else:
        msg = "I didn't quite catch that. Please say 'yes' or 'no' if you were responding to my previous question, or feel free to ask a new question."
        state.next_step = END

    state.response = msg
    state.messages.append(AIMessage(content=msg))
    return state

# --------------------------------------------------------------------------- #
# Build & compile graph
# --------------------------------------------------------------------------- #

graph = StateGraph(ChatState)

graph.add_node("process_user_input", process_user_input)
graph.add_node("check_query_type", check_query_type)
graph.add_node("generate_non_monument_response", generate_non_monument_response)
graph.add_node("send_otp", send_otp_step)
graph.add_node("process_otp_input", process_otp_input)
graph.add_node("final_confirmation", final_confirmation)
graph.add_node("end_conversation", end_conversation)
graph.add_node("initial_greeting_node", initial_greeting_node)
graph.add_node("handle_user_choice", handle_user_choice)
graph.add_node("generate_monument_response", generate_monument_response)

graph.set_entry_point("process_user_input")

graph.add_conditional_edges(
    "process_user_input",
    lambda s: s.next_step,
    {
        "send_otp": "send_otp",
        "process_otp_input": "process_otp_input",
        "check_query_type": "check_query_type",
        "initial_greeting_node": "initial_greeting_node",
        "handle_user_choice": "handle_user_choice",
        END: END,
    },
)
graph.add_conditional_edges(
    "check_query_type",
    lambda s: s.next_step,
    {
        "generate_monument_response": "generate_monument_response",
        "generate_non_monument_response": "generate_non_monument_response",
    },
)
graph.add_edge("generate_non_monument_response", END)
graph.add_edge("send_otp", END)
graph.add_conditional_edges(
    "process_otp_input",
    lambda s: s.next_step,
    {"final_confirmation": "final_confirmation", END: END, "end_conversation": "end_conversation"},
)
graph.add_edge("final_confirmation", END)
graph.add_edge("end_conversation", END)
graph.add_edge("initial_greeting_node", END)
graph.add_edge("handle_user_choice", END)
graph.add_edge("generate_monument_response", END)

compiled_chat_graph = graph.compile()

__all__ = ["compiled_chat_graph", "ChatState"]
