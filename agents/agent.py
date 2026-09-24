from langchain_ollama import ChatOllama


MODEL_NAME = "llama3.2:3b"


llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0
)


def create_agent(tools):
    """
    Create an LLM that can use the provided tools.
    """

    return llm.bind_tools(
        tools,
        tool_choice="auto"
    )