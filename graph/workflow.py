import os
from typing import TypedDict, Optional

from dotenv import load_dotenv

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessage
)

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from agents.agent import llm

from tools.web_search import web_search
from rag.rag import search_documents


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

SENDER_NAME = os.getenv(
    "SENDER_NAME",
    "User"
)


# ============================================================
# STATE
# ============================================================

class AgentState(TypedDict):
    messages: list[BaseMessage]
    route: str

    email_recipient: Optional[str]
    email_subject: Optional[str]
    email_body: Optional[str]


# ============================================================
# ROUTER PROMPT
# ============================================================

ROUTER_PROMPT = """
You are a routing classifier for an AI assistant.

The conversation may contain several previous messages.
Use the conversation context to understand what the user's
latest message refers to.

Classify the user's latest request into exactly ONE category:

DOCUMENT
WEB
EMAIL
GENERAL


DOCUMENT:
Choose DOCUMENT when the user asks about information
contained in their uploaded document or project.

This also includes follow-up questions referring to a
previous document question.

Examples:
- What are the hardware requirements in my project?
- How many landmarks are identified?
- What technologies are mentioned in my PDF?
- Explain the methodology used in my project.
- Explain the first requirement in more detail.
- When was it completed?


WEB:
Choose WEB when the user needs current, recent, live,
or internet-based information.

This also includes follow-up questions referring to
a previous web-search question.

Examples:
- What are the latest developments in generative AI?
- What happened this week?
- Which company released it?
- What is the latest version?


EMAIL:
Choose EMAIL when the user wants to compose, draft,
or send an email.

This also includes follow-up instructions about an
email draft.

Examples:
- Send an email to Rahul saying the meeting is tomorrow.
- Email Priya about the project update.
- Change the subject to project update.
- Send that email.


GENERAL:
Choose GENERAL for normal questions that do not require
the uploaded document, current internet information,
or email automation.

Examples:
- What is machine learning?
- Explain overfitting.
- What is a neural network?
- Explain that concept again.

Return ONLY one word:

DOCUMENT
WEB
EMAIL
GENERAL
"""


# ============================================================
# ROUTER
# ============================================================

def router_node(state: AgentState):

    messages = state["messages"]

    conversation_context = messages[-8:]

    context_text = []

    for message in conversation_context:

        if message.type == "human":
            role = "User"

        elif message.type == "ai":
            role = "Assistant"

        else:
            continue

        context_text.append(
            f"{role}: {message.content}"
        )

    context_text = "\n".join(
        context_text
    )

    response = llm.invoke(
        [
            SystemMessage(
                content=ROUTER_PROMPT
            ),
            HumanMessage(
                content=f"""
Conversation context:

{context_text}

Classify the user's latest message.
"""
            )
        ]
    )

    route = response.content.strip().upper()

    if "DOCUMENT" in route:

        route = "DOCUMENT"

    elif "WEB" in route:

        route = "WEB"

    elif "EMAIL" in route:

        route = "EMAIL"

    else:

        route = "GENERAL"

    print(
        f"\nRouter → {route}"
    )

    return {
        "route": route
    }


# ============================================================
# ROUTE DECISION
# ============================================================

def route_decision(state: AgentState):

    route = state["route"]

    if route == "DOCUMENT":
        return "document"

    elif route == "WEB":
        return "web"

    elif route == "EMAIL":
        return "email"

    return "general"


# ============================================================
# CONVERSATION CONTEXT HELPER
# ============================================================

def get_conversation_context(
    state: AgentState,
    number_of_messages: int = 6
) -> str:

    messages = state["messages"]

    recent_messages = messages[
        -number_of_messages:
    ]

    context = []

    for message in recent_messages:

        if message.type == "human":

            context.append(
                f"User: {message.content}"
            )

        elif message.type == "ai":

            context.append(
                f"Assistant: {message.content}"
            )

    return "\n".join(
        context
    )


# ============================================================
# DOCUMENT NODE
# ============================================================

def document_node(state: AgentState):

    user_message = state["messages"][-1].content

    conversation_context = (
        get_conversation_context(
            state
        )
    )

    # --------------------------------------------------------
    # Convert follow-up question into standalone query
    # --------------------------------------------------------

    query_response = llm.invoke(
        [
            SystemMessage(
                content="""
You are a document-search query reformulator.

Rewrite the user's latest question into a standalone
search query that can be used to retrieve relevant
information from an uploaded PDF.

Use the conversation context to resolve references such as:

- it
- this
- that
- the first one
- the second requirement
- this project
- the document
- that section

Do not answer the question.

Return ONLY the standalone search query.
"""
            ),
            HumanMessage(
                content=f"""
Conversation:

{conversation_context}

Latest user question:

{user_message}
"""
            )
        ]
    )

    search_query = (
        query_response.content.strip()
    )

    if not search_query:

        search_query = user_message

    print(
        f"Document search query → {search_query}"
    )

    # --------------------------------------------------------
    # RAG retrieval
    # --------------------------------------------------------

    result = search_documents.invoke(
        search_query
    )

    # --------------------------------------------------------
    # Final answer
    # --------------------------------------------------------

    response = llm.invoke(
        [
            SystemMessage(
                content="""
You are answering a user's question using their
uploaded document.

Use the retrieved document content as the primary source.

Use the conversation history to understand follow-up
questions and references.

Rules:
- Preserve exact numbers from the document.
- Preserve names and terminology from the document.
- Do not claim information is from the document if it
  was not present in the retrieved content.
- You may provide additional general knowledge when useful,
  but clearly distinguish it from information found in
  the document.
- Do not invent information.
- Answer naturally and directly.
"""
            ),
            HumanMessage(
                content=f"""
Conversation context:

{conversation_context}

Latest user question:

{user_message}

Retrieved document content:

{result}
"""
            )
        ]
    )

    return {
        "messages": [
            response
        ]
    }


# ============================================================
# WEB NODE
# ============================================================

def web_node(state: AgentState):

    user_message = state["messages"][-1].content

    conversation_context = (
        get_conversation_context(
            state
        )
    )

    # --------------------------------------------------------
    # Create context-aware web search query
    # --------------------------------------------------------

    query_response = llm.invoke(
        [
            SystemMessage(
                content="""
You are a web-search query reformulator.

Rewrite the latest user question into a standalone
web-search query.

Use the conversation context to resolve references such as:
- it
- this
- that
- they
- the company
- the previous topic

Do not answer the question.

Return ONLY the search query.
"""
            ),
            HumanMessage(
                content=f"""
Conversation:

{conversation_context}

Latest user question:

{user_message}
"""
            )
        ]
    )

    search_query = (
        query_response.content.strip()
    )

    if not search_query:

        search_query = user_message

    result = web_search.invoke(
        search_query
    )

    response = llm.invoke(
        [
            SystemMessage(
                content="""
You are answering a user's question using web
search results.

Use the retrieved web results to answer the question.

Use the conversation history when the latest question
depends on earlier context.

Rules:
- Prefer information from the search results.
- Do not pretend information is current if the search
  results do not support it.
- Clearly state uncertainty when the results are insufficient.
- Answer naturally and directly.
"""
            ),
            HumanMessage(
                content=f"""
Conversation context:

{conversation_context}

Latest user question:

{user_message}

Web search results:

{result}
"""
            )
        ]
    )

    return {
        "messages": [
            response
        ]
    }


# ============================================================
# GENERAL NODE
# ============================================================

def general_node(state: AgentState):

    user_message = state["messages"][-1].content

    conversation_context = (
        get_conversation_context(
            state
        )
    )

    response = llm.invoke(
        [
            SystemMessage(
                content="""
You are a helpful conversational AI assistant.

Use the conversation history to understand references
and follow-up questions.

Answer naturally and clearly.
"""
            ),
            HumanMessage(
                content=f"""
Conversation context:

{conversation_context}

Latest user question:

{user_message}
"""
            )
        ]
    )

    return {
        "messages": [
            response
        ]
    }


# ============================================================
# EMAIL NODE
# ============================================================

def email_node(state: AgentState):

    user_message = state["messages"][-1].content

    conversation_context = (
        get_conversation_context(
            state
        )
    )

    response = llm.invoke(
        [
            SystemMessage(
                content=f"""
You are an email drafting assistant.

The configured sender name is:

{SENDER_NAME}

Use the conversation context to understand follow-up
instructions about an email.

Extract the email information from the user's request.

Return the result EXACTLY in this format:

RECIPIENT:
<email address or MISSING>

SUBJECT:
<email subject>

BODY:
<complete email body>

Rules:

- If the recipient email address is not provided and
  cannot be determined from conversation context,
  write MISSING.
- If the user refers to an existing email draft, preserve
  the recipient unless the user explicitly changes it.
- Create a suitable subject if the user did not provide one.
- Write a professional but natural email.
- Stay faithful to the user's requested message.
- Do not add claims, promises, deadlines, or facts that
  the user did not provide.
- Do not infer the recipient's name from their email address.
- Do not use placeholders such as [Your Name].
- If a sign-off is appropriate, use:
  {SENDER_NAME}
- Do not invent an email address.
- Do not send the email.
"""
            ),
            HumanMessage(
                content=f"""
Conversation context:

{conversation_context}

Latest user request:

{user_message}
"""
            )
        ]
    )

    draft = response.content.strip()

    # --------------------------------------------------------
    # Recipient
    # --------------------------------------------------------

    recipient = ""

    if "RECIPIENT:" in draft:

        recipient_part = draft.split(
            "RECIPIENT:",
            1
        )[1]

        if "SUBJECT:" in recipient_part:

            recipient = recipient_part.split(
                "SUBJECT:",
                1
            )[0].strip()

    # --------------------------------------------------------
    # Subject
    # --------------------------------------------------------

    subject = ""

    if "SUBJECT:" in draft:

        subject_part = draft.split(
            "SUBJECT:",
            1
        )[1]

        if "BODY:" in subject_part:

            subject = subject_part.split(
                "BODY:",
                1
            )[0].strip()

    # --------------------------------------------------------
    # Body
    # --------------------------------------------------------

    body = ""

    if "BODY:" in draft:

        body = draft.split(
            "BODY:",
            1
        )[1].strip()

    # --------------------------------------------------------
    # Return draft
    # --------------------------------------------------------

    return {
        "messages": [
            AIMessage(
                content="Email draft prepared."
            )
        ],
        "email_recipient": recipient,
        "email_subject": subject,
        "email_body": body
    }


# ============================================================
# BUILD GRAPH
# ============================================================

workflow = StateGraph(
    AgentState
)

workflow.add_node(
    "router",
    router_node
)

workflow.add_node(
    "document",
    document_node
)

workflow.add_node(
    "web",
    web_node
)

workflow.add_node(
    "general",
    general_node
)

workflow.add_node(
    "email",
    email_node
)


workflow.add_edge(
    START,
    "router"
)


workflow.add_conditional_edges(
    "router",
    route_decision,
    {
        "document": "document",
        "web": "web",
        "email": "email",
        "general": "general"
    }
)


workflow.add_edge(
    "document",
    END
)

workflow.add_edge(
    "web",
    END
)

workflow.add_edge(
    "general",
    END
)

workflow.add_edge(
    "email",
    END
)


app = workflow.compile()