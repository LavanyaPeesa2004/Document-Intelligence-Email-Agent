import hashlib
from pathlib import Path

import streamlit as st

from langchain_core.messages import (
    HumanMessage,
    AIMessage
)

from graph.workflow import app
from rag.rag import ingest_document
from tools.email import send_email


# ============================================================
# CONFIGURATION
# ============================================================

UPLOAD_FOLDER = Path(
    "uploaded_documents"
)

UPLOAD_FOLDER.mkdir(
    exist_ok=True
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=(
        "Document Intelligence & Email Automation Agent"
    ),
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "uploaded_files" not in st.session_state:

    st.session_state.uploaded_files = []


if "pending_email" not in st.session_state:

    st.session_state.pending_email = None


# ============================================================
# TITLE
# ============================================================

st.title(
    "🤖 Document Intelligence & Email Automation Agent"
)

st.caption(
    "Upload documents, ask questions, search the web, "
    "and automate email tasks."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "📄 Document Upload"
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        help=(
            "Upload a PDF document for "
            "document-based questions."
        )
    )

    if uploaded_file is not None:

        file_bytes = (
            uploaded_file.getvalue()
        )

        file_hash = hashlib.sha256(
            file_bytes
        ).hexdigest()

        original_name = Path(
            uploaded_file.name
        ).name

        saved_name = (
            f"{file_hash[:12]}_{original_name}"
        )

        saved_path = (
            UPLOAD_FOLDER / saved_name
        )

        if not saved_path.exists():

            with open(
                saved_path,
                "wb"
            ) as file:

                file.write(
                    file_bytes
                )

        if (
            original_name
            not in st.session_state.uploaded_files
        ):

            with st.spinner(
                "Processing document..."
            ):

                result = ingest_document(
                    str(saved_path)
                )

            st.session_state.uploaded_files.append(
                original_name
            )

            st.success(
                result
            )

        else:

            st.info(
                f"'{original_name}' is already loaded."
            )

    st.divider()

    st.subheader(
        "📚 Loaded Documents"
    )

    if st.session_state.uploaded_files:

        for filename in (
            st.session_state.uploaded_files
        ):

            st.write(
                f"✅ {filename}"
            )

    else:

        st.caption(
            "No documents uploaded yet."
        )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in (
    st.session_state.messages
):

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# PENDING EMAIL
# ============================================================

if st.session_state.pending_email:

    email_data = (
        st.session_state.pending_email
    )

    st.divider()

    st.subheader(
        "📧 Email Draft"
    )

    st.write(
        f"**To:** {email_data['recipient']}"
    )

    st.write(
        f"**Subject:** {email_data['subject']}"
    )

    st.markdown(
        "**Message:**"
    )

    st.text(
        email_data["body"]
    )

    col1, col2 = st.columns(2)

    with col1:

        send_clicked = st.button(
            "✅ Send Email",
            use_container_width=True
        )

    with col2:

        cancel_clicked = st.button(
            "❌ Cancel",
            use_container_width=True
        )

    if send_clicked:

        with st.spinner(
            "Sending email..."
        ):

            result = send_email.invoke(
                {
                    "recipient": email_data[
                        "recipient"
                    ],
                    "subject": email_data[
                        "subject"
                    ],
                    "body": email_data[
                        "body"
                    ]
                }
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"📧 {result}"
            }
        )

        st.session_state.pending_email = None

        st.rerun()

    if cancel_clicked:

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "📧 Email cancelled. "
                    "The draft was not sent."
                )
            }
        )

        st.session_state.pending_email = None

        st.rerun()


# ============================================================
# CHAT INPUT
# ============================================================

user_input = st.chat_input(
    "Ask me anything..."
)


if user_input:

    # --------------------------------------------------------
    # Add current user message to UI history
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            user_input
        )

    # --------------------------------------------------------
    # Convert UI history into LangChain messages
    # --------------------------------------------------------

    conversation_messages = []

    for message in (
        st.session_state.messages
    ):

        if message["role"] == "user":

            conversation_messages.append(
                HumanMessage(
                    content=message["content"]
                )
            )

        elif message["role"] == "assistant":

            conversation_messages.append(
                AIMessage(
                    content=message["content"]
                )
            )

    # --------------------------------------------------------
    # Run agent
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Thinking..."
        ):

            try:

                result = app.invoke(
                    {
                        "messages": conversation_messages,
                        "route": "",
                        "email_recipient": None,
                        "email_subject": None,
                        "email_body": None
                    }
                )

                route = result.get(
                    "route",
                    ""
                )

                # ====================================================
                # EMAIL
                # ====================================================

                if route == "EMAIL":

                    recipient = result.get(
                        "email_recipient"
                    )

                    subject = result.get(
                        "email_subject"
                    )

                    body = result.get(
                        "email_body"
                    )

                    if not recipient:

                        answer = (
                            "I need the recipient's "
                            "email address before I can "
                            "prepare the email."
                        )

                        st.markdown(
                            answer
                        )

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer
                            }
                        )

                    else:

                        st.session_state.pending_email = {
                            "recipient": recipient,
                            "subject": subject,
                            "body": body
                        }

                        answer = (
                            "📧 I prepared the email "
                            "draft for your review below."
                        )

                        st.markdown(
                            answer
                        )

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer
                            }
                        )

                        st.rerun()

                # ====================================================
                # NORMAL RESPONSE
                # ====================================================

                else:

                    messages = result[
                        "messages"
                    ]

                    answer = (
                        "I could not generate "
                        "a response."
                    )

                    for message in reversed(
                        messages
                    ):

                        if (
                            message.type == "ai"
                            and message.content
                        ):

                            answer = (
                                message.content
                            )

                            break

                    st.markdown(
                        answer
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

            except Exception as e:

                st.error(
                    f"An error occurred: {e}"
                )