from pathlib import Path
import hashlib

from langchain_ollama import OllamaEmbeddings
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_FOLDER = Path("chroma_db")
EMBEDDING_MODEL = "nomic-embed-text"


# ============================================================
# EMBEDDING MODEL
# ============================================================

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)


# ============================================================
# CHROMA VECTOR STORE
# ============================================================

vector_store = Chroma(
    persist_directory=str(CHROMA_FOLDER),
    embedding_function=embeddings
)


# ============================================================
# FILE HASH
# ============================================================

def calculate_file_hash(file_path: Path) -> str:
    """
    Calculate a SHA-256 hash for a file.

    This allows us to identify whether the exact same
    document has already been ingested.
    """

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        while True:

            data = file.read(1024 * 1024)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


# ============================================================
# CHECK WHETHER DOCUMENT ALREADY EXISTS
# ============================================================

def document_already_ingested(file_hash: str) -> bool:
    """
    Check whether a document with the same file hash
    already exists in Chroma.
    """

    try:

        existing = vector_store.get(
            where={
                "file_hash": file_hash
            },
            limit=1
        )

        return bool(
            existing.get("ids")
        )

    except Exception:

        return False


# ============================================================
# DOCUMENT INGESTION
# ============================================================

def ingest_document(file_path: str) -> str:
    """
    Load a PDF, split it into chunks, create embeddings,
    and add the chunks to Chroma.

    Duplicate documents are skipped.
    """

    pdf_path = Path(file_path).expanduser()

    # --------------------------------------------------------
    # Validate file
    # --------------------------------------------------------

    if not pdf_path.exists():

        return (
            f"File not found: {pdf_path}"
        )

    if not pdf_path.is_file():

        return (
            f"The provided path is not a file: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":

        return (
            "Only PDF files are currently supported."
        )

    try:

        print("\n" + "=" * 60)
        print("DOCUMENT INGESTION")
        print("=" * 60)

        print(
            f"Loading: {pdf_path.name}"
        )

        # ----------------------------------------------------
        # Calculate file hash
        # ----------------------------------------------------

        file_hash = calculate_file_hash(
            pdf_path
        )

        # ----------------------------------------------------
        # Duplicate check
        # ----------------------------------------------------

        if document_already_ingested(file_hash):

            print(
                "Document already exists in Chroma."
            )

            print("=" * 60)

            return (
                f"'{pdf_path.name}' is already "
                "ingested. No duplicate chunks were added."
            )

        # ----------------------------------------------------
        # Load PDF
        # ----------------------------------------------------

        loader = PyPDFLoader(
            str(pdf_path)
        )

        documents = loader.load()

        if not documents:

            return (
                "No readable content was found "
                "in the PDF."
            )

        print(
            f"Pages loaded: {len(documents)}"
        )

        # ----------------------------------------------------
        # Split into chunks
        # ----------------------------------------------------

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = text_splitter.split_documents(
            documents
        )

        if not chunks:

            return (
                "Could not create document chunks."
            )

        print(
            f"Chunks created: {len(chunks)}"
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        source_id = str(
            pdf_path.resolve()
        )

        for chunk in chunks:

            chunk.metadata["document_name"] = (
                pdf_path.name
            )

            chunk.metadata["source"] = (
                source_id
            )

            chunk.metadata["file_hash"] = (
                file_hash
            )

        # ----------------------------------------------------
        # Unique IDs
        # ----------------------------------------------------

        ids = [
            f"{file_hash}_{index}"
            for index in range(len(chunks))
        ]

        # ----------------------------------------------------
        # Add to Chroma
        # ----------------------------------------------------

        vector_store.add_documents(
            documents=chunks,
            ids=ids
        )

        print(
            "Document added to Chroma successfully."
        )

        print("=" * 60)

        return (
            f"Successfully ingested "
            f"'{pdf_path.name}'. "
            f"Created {len(chunks)} chunks."
        )

    except Exception as e:

        return (
            f"Document ingestion failed: {e}"
        )


# ============================================================
# DOCUMENT SEARCH TOOL
# ============================================================

@tool
def search_documents(query: str) -> str:
    """
    Search uploaded PDF documents for relevant information.
    """

    try:

        results = vector_store.similarity_search(
            query,
            k=4
        )

        if not results:

            return (
                "No relevant information found "
                "in the uploaded documents."
            )

        output = []

        for index, document in enumerate(
            results,
            1
        ):

            document_name = document.metadata.get(
                "document_name",
                "Unknown"
            )

            page = document.metadata.get(
                "page",
                "Unknown"
            )

            output.append(
                f"""
Document Result {index}

Document: {document_name}
Page: {page}

Content:
{document.page_content}
"""
            )

        return "\n".join(
            output
        )

    except Exception as e:

        return (
            f"Document search failed: {e}"
        )