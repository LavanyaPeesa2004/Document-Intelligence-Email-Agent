from ddgs import DDGS
from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """
    Search the web for current or factual information.
    """

    try:
        with DDGS() as ddgs:

            results = list(
                ddgs.text(
                    query,
                    max_results=5
                )
            )

        if not results:
            return "No web results found."

        output = []

        for i, result in enumerate(results, 1):

            output.append(
                f"""
Result {i}
Title: {result.get('title', '')}
Description: {result.get('body', '')}
URL: {result.get('href', '')}
"""
            )

        return "\n".join(output)

    except Exception as e:

        return f"Web search failed: {e}"