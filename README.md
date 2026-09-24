# 🤖 Document Intelligence & Email Automation Agent

A GenAI-powered intelligent assistant that combines **document intelligence, web search, conversational context, and email automation** into a single application.

The agent automatically understands the user's request and routes it to the appropriate workflow without requiring the user to manually select a mode.

---

## 🚀 Project Overview

The **Document Intelligence & Email Automation Agent** is a multi-capability AI assistant built using **LangChain, LangGraph, Ollama, RAG, Chroma, Streamlit, web search, and Gmail SMTP**.

The application can:

- Upload PDF documents directly from the user's computer
- Answer questions using Retrieval-Augmented Generation (RAG)
- Understand follow-up questions using conversation context
- Search the web for current information
- Answer general AI and technical questions
- Draft emails from natural-language instructions
- Require human approval before sending emails
- Send approved emails through Gmail SMTP

Instead of forcing the user to select a specific mode, the system automatically routes each request to the appropriate workflow.

---

## ✨ Features

### 📄 Dynamic PDF Upload

Users can upload PDF documents directly from their computer through the Streamlit interface.

The system processes the uploaded document through the following pipeline:

```text
PDF Upload
    ↓
PyPDFLoader
    ↓
Text Extraction
    ↓
Chunking
    ↓
Nomic Embeddings
    ↓
Chroma Vector Database
    ↓
Semantic Retrieval
````

The uploaded PDF then becomes available for document-based questions.

---

### 🔎 Retrieval-Augmented Generation (RAG)

The project uses RAG to answer questions from uploaded documents.

The RAG pipeline is:

```text
User Question
      ↓
Document Search
      ↓
Chroma Similarity Search
      ↓
Relevant Document Chunks
      ↓
Llama 3.2
      ↓
Context-Aware Answer
```

The application uses:

* `PyPDFLoader` for PDF loading
* `RecursiveCharacterTextSplitter` for chunking
* `nomic-embed-text` for embeddings
* `Chroma` for vector storage
* `Llama 3.2:3b` for answer generation

---

### 🧠 Context-Aware Document Questions

The agent can understand follow-up questions instead of treating every question independently.

Example:

```text
User:
What does this PDF contain?

AI:
This PDF contains a certificate of completion...

User:
Who completed it?

AI:
The certificate was completed by...

User:
When was it completed?

AI:
It was completed on November 22, 2024.
```

The same approach works with project-related questions:

```text
User:
What are the hardware requirements?

AI:
The hardware requirements include...

User:
Explain the first requirement in detail.

AI:
The first requirement is the processor...
```

The system uses previous conversation context to understand references such as:

* it
* this
* that
* the first one
* the second requirement
* the document
* the previous question

---

### 🌐 Web Search

When the user asks for current, recent, or internet-based information, the agent automatically routes the request to the web-search workflow.

Example:

```text
User:
What are the latest developments in Generative AI?

Router:
WEB

Web Search
    ↓
Search Results
    ↓
Llama 3.2
    ↓
Final Answer
```

The project uses **DDGS** for web search.

---

### 📧 Email Automation

The agent can draft and send emails using natural-language instructions.

Example:

```text
Write an email to recruiter@example.com
about my project progress.
```

The system extracts:

* Recipient
* Subject
* Email body

and generates a professional email draft.

---

### 👤 Human Approval Before Email Sending

The application does not immediately send AI-generated emails.

The generated email is first displayed in the browser:

```text
📧 Email Draft

To: recruiter@example.com
Subject: Project Progress Update

Message:
...

✅ Send Email
❌ Cancel
```

The email is sent only after the user explicitly clicks:

```text
✅ Send Email
```

This creates a human-in-the-loop approval step before performing the external action.

---

### 🧠 Intelligent Request Routing

The project uses **LangGraph** to route user requests into four categories:

```text
DOCUMENT
WEB
EMAIL
GENERAL
```

The routing is performed automatically using the LLM.

The user does not have to manually select a mode.

Example:

```text
"What are the hardware requirements in my project?"
                    ↓
               DOCUMENT
                    ↓
                   RAG
```

```text
"What are the latest AI developments?"
                    ↓
                   WEB
                    ↓
               Web Search
```

```text
"Write an email to my mentor about my project."
                    ↓
                  EMAIL
                    ↓
              Email Draft
```

```text
"What is overfitting?"
                    ↓
                 GENERAL
                    ↓
              Llama 3.2
```

---

# 🏗️ System Architecture

```text
                         ┌─────────────────────┐
                         │     Streamlit UI    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    LangGraph        │
                         │  Intelligent Router │
                         └──────────┬──────────┘
                                    │
             ┌──────────────────────┼──────────────────────┐
             │                      │                      │
             ▼                      ▼                      ▼
     ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
     │   DOCUMENT    │      │      WEB      │      │     EMAIL     │
     │      RAG      │      │    SEARCH     │      │   AUTOMATION  │
     └───────┬───────┘      └───────┬───────┘      └───────┬───────┘
             │                      │                      │
             ▼                      ▼                      ▼
     ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
     │    Chroma     │      │     DDGS      │      │   Gmail SMTP  │
     │ Vector Store  │      │ Web Search    │      │ Email Sending │
     └───────┬───────┘      └───────────────┘      └───────────────┘
             │
             ▼
     ┌─────────────────────┐
     │     Llama 3.2:3b    │
     │    Local LLM         │
     └─────────────────────┘
```

---

# 🛠️ Technologies Used

| Technology                        | Purpose                                        |
| --------------------------------- | ---------------------------------------------- |
| Python                            | Core application development                   |
| Streamlit                         | User interface                                 |
| LangChain                         | LLM and tool integration                       |
| LangGraph                         | Workflow orchestration and intelligent routing |
| Ollama                            | Local LLM execution                            |
| Llama 3.2:3b                      | Language model                                 |
| Nomic Embed Text                  | Document embeddings                            |
| Chroma                            | Vector database                                |
| PyPDF                             | PDF document loading                           |
| Recursive Character Text Splitter | Document chunking                              |
| DDGS                              | Web search                                     |
| Gmail SMTP                        | Email delivery                                 |
| python-dotenv                     | Environment variable management                |

---

# 📁 Project Structure

```text
Document-Intelligence-Email-Agent/
│
├── agents/
│   ├── __init__.py
│   └── agent.py
│
├── graph/
│   ├── __init__.py
│   └── workflow.py
│
├── rag/
│   ├── __init__.py
│   └── rag.py
│
├── tools/
│   ├── __init__.py
│   ├── email.py
│   └── web_search.py
│
├── app.py
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

Local runtime data such as the following are intentionally excluded from GitHub:

```text
chroma_db/
uploaded_documents/
.env
documents/*.pdf
```

---

# ⚙️ Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/LavanyaPeesa2004/Document-Intelligence-Email-Agent.git
```

Move into the project directory:

```bash
cd Document-Intelligence-Email-Agent
```

---

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🦙 Ollama Setup

This project uses Ollama to run the language model locally.

Install Ollama and pull the required models.

### Llama 3.2

```bash
ollama pull llama3.2:3b
```

### Nomic Embeddings

```bash
ollama pull nomic-embed-text
```

Make sure Ollama is running before starting the application.

---

# 🔐 Environment Configuration

Create a `.env` file in the project root.

Example:

```env
EMAIL_ADDRESS=your_email@gmail.com
EMAIL_APP_PASSWORD=your_gmail_app_password
SENDER_NAME=Your Name
```

The application reads these values using `python-dotenv`.

### Important

Do not upload your real `.env` file to GitHub.

The `.gitignore` file excludes:

```text
.env
```

A safe example configuration can be provided using:

```text
.env.example
```

---

# ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

The interface allows you to:

1. Upload a PDF
2. Ask questions about the document
3. Ask general questions
4. Search the web
5. Draft emails
6. Review email drafts
7. Approve or cancel email sending

---

# 🧪 Example Usage

## Example 1 — Document Question

Upload a PDF from your computer.

Then ask:

```text
What does this PDF contain?
```

The application automatically routes the request to the document workflow.

---

## Example 2 — Follow-Up Question

After asking about the document:

```text
Who completed it?
```

You can continue with:

```text
When was it completed?
```

The agent uses conversation context to understand what "it" refers to.

---

## Example 3 — Project Question

Upload a project report and ask:

```text
What are the hardware requirements?
```

Then:

```text
Explain the first requirement in detail.
```

The agent reformulates the follow-up question for document retrieval before generating the final answer.

---

## Example 4 — Web Search

Ask:

```text
What are the latest developments in Generative AI?
```

The agent automatically routes the request to the web-search workflow.

---

## Example 5 — General AI Question

Ask:

```text
What is overfitting in machine learning?
```

The agent routes the request to the general LLM workflow.

---

## Example 6 — Email Automation

Ask:

```text
Write an email to recruiter@example.com
about my project progress.
```

The agent generates an email draft.

The browser then displays:

```text
📧 Email Draft

To: recruiter@example.com
Subject: ...

Message:
...

✅ Send Email
❌ Cancel
```

The email is sent only after the user explicitly approves it.

---

# 🔄 End-to-End Workflow

```text
                    User
                     │
                     ▼
              Streamlit Interface
                     │
                     ▼
              Natural Language Query
                     │
                     ▼
             Intelligent Router
                     │
       ┌─────────────┼─────────────┐
       │             │             │
       ▼             ▼             ▼
   DOCUMENT         WEB          EMAIL
       │             │             │
       ▼             ▼             ▼
      RAG        Web Search    Email Draft
       │             │             │
       ▼             │             ▼
    Chroma           │        Human Approval
       │             │             │
       └─────────────┼─────────────┘
                     │
                     ▼
                Llama 3.2
                     │
                     ▼
               Final Response
```

---

# 🛡️ Security

The project keeps sensitive information outside the source code.

The following files and directories are excluded from version control:

```text
.env
chroma_db/
uploaded_documents/
documents/*.pdf
```

Email credentials are loaded from environment variables rather than being hard-coded into the application.

The email workflow also requires explicit user approval before sending.

---

# 🧠 AI Agent Design

The project uses a graph-based architecture instead of placing all functionality inside one large Python script.

The main LangGraph workflow contains separate nodes for:

```text
Router
Document
Web
Email
General
```

This separation makes the application easier to maintain and extend.

The routing decision is made automatically from the user's request and conversation context.

---

# 📚 RAG Design

The RAG implementation follows a standard document intelligence pipeline:

```text
PDF
 ↓
Document Loader
 ↓
Text Extraction
 ↓
Recursive Chunking
 ↓
Embedding Generation
 ↓
Chroma Vector Store
 ↓
Semantic Similarity Search
 ↓
Retrieved Context
 ↓
Llama 3.2
 ↓
Answer
```

The system also stores metadata such as:

* Document name
* Source
* Page number
* File hash

This allows retrieved chunks to retain information about their source document.

---

# 📧 Email Automation Design

The email automation workflow is:

```text
Natural Language Request
          ↓
Email Intent Detection
          ↓
Recipient Extraction
          ↓
Subject Generation
          ↓
Email Body Generation
          ↓
Draft Display
          ↓
User Approval
       ↙     ↘
    Cancel    Send
                 ↓
             Gmail SMTP
```

The human approval step prevents the agent from automatically sending an email without user confirmation.

---

# 📌 Current Limitations

This project is currently designed as a **portfolio and learning application**.

Current limitations include:

* PDF is the primary supported document format.
* Conversation memory is session-based.
* Chroma is stored locally.
* The application is designed primarily for a single-user/local environment.
* Gmail SMTP is used for email delivery.
* Production deployment would require stronger authentication, monitoring, persistent storage, and multi-user document isolation.
* Web search quality depends on the available search results.
* Local LLM performance depends on available system hardware.

---

# 🔮 Future Improvements

Potential future improvements include:

* Multi-user authentication
* Persistent conversation memory
* Multi-user document isolation
* Document deletion and management
* DOCX and other document format support
* Improved document citations
* Source-aware answer display
* Cloud vector database
* Production email provider integration
* Docker deployment
* FastAPI backend
* Advanced agent planning
* More tools and external integrations
* Cloud deployment
* Application monitoring and logging

---

# 🎯 Key Learning Outcomes

This project demonstrates practical implementation of:

* Generative AI
* AI Agents
* Retrieval-Augmented Generation (RAG)
* Vector databases
* Embeddings
* LangChain
* LangGraph
* LLM-based routing
* Tool integration
* Conversational AI
* Web search integration
* Email automation
* Human-in-the-loop workflows
* Streamlit application development
* Environment variable management
* Local LLM deployment

---

# 💡 Why This Project?

Instead of building a simple chatbot, this project combines multiple real-world AI capabilities into one workflow.

The system demonstrates how an AI assistant can:

```text
Understand
   ↓
Route
   ↓
Retrieve / Search / Generate
   ↓
Take an Action
   ↓
Ask for Human Approval
   ↓
Execute the Action
```

This makes the project a practical demonstration of **Generative AI, RAG, AI agents, workflow orchestration, and automation**.

---

# 👩‍💻 Author

## Lavanya Peesa

B.Tech — Data Science

GitHub:

[https://github.com/LavanyaPeesa2004](https://github.com/LavanyaPeesa2004)

---

# ⭐ Project Highlights

```text
📄 Dynamic PDF Upload
🔎 Retrieval-Augmented Generation
🧠 Intelligent LLM Routing
🌐 Web Search
💬 Conversational Context
📧 Email Automation
👤 Human-in-the-Loop Approval
🦙 Local LLM with Ollama
🗄️ Chroma Vector Database
🖥️ Streamlit Application
🔐 Environment-Based Secret Management
```

---

