from pathlib import Path

from langchain_core.messages import HumanMessage

from graph.workflow import app
from rag.rag import ingest_document


# ============================================================
# AI AGENT
# ============================================================

def ai_agent(query: str):

    result = app.invoke(
        {
            "messages": [
                HumanMessage(content=query)
            ],
            "route": ""
        }
    )

    messages = result["messages"]

    for message in reversed(messages):

        if (
            message.type == "ai"
            and message.content
        ):

            return message.content

    return "I could not generate a response."


# ============================================================
# UPLOAD COMMAND
# ============================================================

def handle_upload(command: str):

    file_path = command[len("/upload"):].strip()

    # --------------------------------------------------------
    # If user entered only /upload
    # --------------------------------------------------------

    if not file_path:

        file_path = input(
            "\nEnter the path to the PDF: "
        ).strip()

    # --------------------------------------------------------
    # Remove surrounding quotes
    # --------------------------------------------------------

    if (
        len(file_path) >= 2
        and file_path[0] == file_path[-1]
        and file_path[0] in ["\"", "'"]
    ):

        file_path = file_path[1:-1]

    # --------------------------------------------------------
    # Validate path
    # --------------------------------------------------------

    path = Path(
        file_path
    ).expanduser()

    if not path.exists():

        print(
            "\nAI:"
        )

        print(
            f"File not found: {path}"
        )

        return

    # --------------------------------------------------------
    # Ingest document
    # --------------------------------------------------------

    result = ingest_document(
        str(path)
    )

    print(
        "\nAI:"
    )

    print(
        result
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "Document Intelligence & Email Automation Agent"
    )

    print(
        "LangGraph Agent"
    )

    print(
        "Type 'exit' to quit"
    )

    print(
        "Use '/upload <pdf_path>' to upload a document"
    )

    print("=" * 60)

    while True:

        user_input = input(
            "\nYou: "
        ).strip()

        if not user_input:

            continue

        # ----------------------------------------------------
        # Exit
        # ----------------------------------------------------

        if user_input.lower() == "exit":

            print(
                "Goodbye!"
            )

            break

        # ----------------------------------------------------
        # Upload
        # ----------------------------------------------------

        if user_input.lower().startswith(
            "/upload"
        ):

            try:

                handle_upload(
                    user_input
                )

            except Exception as e:

                print(
                    "\nError:"
                )

                print(
                    e
                )

            continue

        # ----------------------------------------------------
        # Normal AI agent request
        # ----------------------------------------------------

        try:

            answer = ai_agent(
                user_input
            )

            print(
                "\nAI:"
            )

            print(
                answer
            )

        except Exception as e:

            print(
                "\nError:"
            )

            print(
                e
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()