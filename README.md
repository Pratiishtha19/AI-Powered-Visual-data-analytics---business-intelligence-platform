# AI-Powered Visual Data Analytics and Business Intelligence

## 📌 Project Overview

This project is developed as part of **Infosys Springboard Internship – Milestone 3**.

The project combines **AI-based image analysis** with a **Retrieval-Augmented Generation (RAG)** based safety assistant. It processes a workplace safety manual, converts the information into searchable vector representations, retrieves relevant information based on the user's question, and generates a simple answer using a local Hugging Face language model.

The project demonstrates the integration of:

* Computer Vision
* YOLOv8
* Document Processing
* Text Chunking
* Hugging Face Embeddings
* FAISS Vector Database
* Retrieval-Augmented Generation (RAG)
* FLAN-T5 Language Model
* Streamlit

---

## 🎯 Objectives

The main objectives of this project are:

1. Detect workplace safety equipment using YOLOv8.
2. Process and extract information from a safety manual.
3. Divide the document into smaller meaningful chunks.
4. Convert text chunks into vector embeddings.
5. Store embeddings in a FAISS vector database.
6. Retrieve relevant safety information based on a user question.
7. Generate a clear answer using the FLAN-T5 model.
8. Build a foundation for an AI-powered workplace safety assistant.

---

## 🏗️ Project Workflow

```text
Safety Manual PDF
       ↓
Text Extraction
       ↓
Text Cleaning / Chunking
       ↓
Hugging Face Embeddings
       ↓
FAISS Vector Database
       ↓
User Safety Question
       ↓
Similarity Search
       ↓
Relevant Safety Information
       ↓
Context + Question
       ↓
FLAN-T5 Model
       ↓
AI Generated Answer
```

---

## 🧠 RAG Implementation

The project uses **Retrieval-Augmented Generation (RAG)**.

Instead of asking the language model to answer from general knowledge, the system first searches the workplace safety manual for relevant information.

### RAG Process

**1. Document Loading**

The safety manual is loaded from:

```text
data/safety_manual.pdf
```

PyMuPDF is used to extract text from the PDF.

**2. Text Chunking**

The extracted text is divided into smaller chunks using:

```text
RecursiveCharacterTextSplitter
```

Configuration:

```text
Chunk Size: 500
Chunk Overlap: 50
```

**3. Embeddings**

The project uses the Hugging Face embedding model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The text chunks are converted into numerical vector representations.

**4. Vector Database**

The embeddings are stored using:

```text
FAISS
```

FAISS allows the system to perform similarity searches efficiently.

**5. Retrieval**

When the user asks a question, the system retrieves the top 3 relevant document chunks.

**6. Generation**

The retrieved information is provided as context to:

```text
google/flan-t5-base
```

The model generates a simple answer based on the retrieved safety information.

---

## 🤖 Computer Vision

The project also contains a YOLOv8-based PPE detection component.

The system can identify workplace safety equipment such as:

* Safety Helmet
* Safety Vest
* Safety Shoes
* Safety Gloves
* Other trained PPE classes

The trained model is stored as:

```text
best.pt
```

---

## 🛠️ Technologies Used

| Technology            | Purpose                   |
| --------------------- | ------------------------- |
| Python                | Core programming language |
| YOLOv8                | PPE detection             |
| Ultralytics           | YOLO implementation       |
| Streamlit             | User interface            |
| PyMuPDF               | PDF text extraction       |
| LangChain             | RAG workflow              |
| Hugging Face          | Embeddings and LLM        |
| FAISS                 | Vector database           |
| Sentence Transformers | Text embeddings           |
| FLAN-T5               | Answer generation         |

---

## 📂 Project Structure

```text
AI-POWERED-VISUAL-DATA-ANALYTICS-AND-BUSINESS-INTELLIGENCE/
│
├── data/
│   └── safety_manual.pdf
│
├── src/
│   ├── documents/
│   │   ├── __init__.py
│   │   ├── chunker.py
│   │   ├── document_loader.py
│   │   ├── metadata.py
│   │   └── text_cleaner.py
│   │
│   └── knowledge_base/
│       ├── __init__.py
│       └── database.py
│
├── best.pt
├── yolov8n.pt
├── milestone3_practice.py
├── ppe_model.py
├── vision.py
├── yolo_model.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

Open the project folder:

```bash
cd AI-POWERED-VISUAL-DATA-ANALYTICS-AND-BUSINESS-INTELLIGENCE
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

If required, install the main RAG dependencies manually:

```bash
pip install transformers sentencepiece torch
pip install langchain-huggingface sentence-transformers
pip install langchain-community langchain-text-splitters
pip install faiss-cpu pymupdf
```

---

## ▶️ Running the RAG Safety Assistant

Run:

```bash
python milestone3_practice.py
```

The program will ask:

```text
Ask a safety question:
```

Example:

```text
What PPE is required in construction areas?
```

The system will:

1. Search the safety manual.
2. Retrieve relevant information.
3. Pass the retrieved information to FLAN-T5.
4. Generate the final answer.

Example output:

```text
AI ANSWER:
A safety helmet is mandatory in production and construction areas.
Safety vest and safety shoes are required according to the workplace safety requirements.
```

---

## 🔐 API Key Requirement

The RAG answer-generation component uses the local Hugging Face model:

```text
google/flan-t5-base
```

Therefore, an **OpenAI API key is not required for the LLM generation**.

The embedding model is also downloaded from Hugging Face:

```text
sentence-transformers/all-MiniLM-L6-v2
```

---

## 📊 Key Features

### 🔹 PPE Detection

Detects workplace safety equipment using YOLOv8.

### 🔹 PDF Processing

Extracts text from the workplace safety manual using PyMuPDF.

### 🔹 Intelligent Chunking

Splits large documents into smaller searchable sections.

### 🔹 Semantic Search

Uses embeddings and FAISS to retrieve relevant information.

### 🔹 RAG-Based Question Answering

Combines retrieved safety information with a language model.

### 🔹 Local AI Model

Uses FLAN-T5 for answer generation without requiring OpenAI credits.

---

## 🚀 Future Enhancements

* Add a complete Streamlit interface for the RAG assistant.
* Support multiple safety manuals.
* Add conversation history.
* Display document sources for every answer.
* Improve PPE detection accuracy.
* Add voice-based safety questions.
* Add multilingual safety assistance.
* Deploy the application as a web application.

---

## 👩‍💻 Author

**Pratishtha Gadwanshi**

Project: **AI-Powered Visual Data Analytics and Business Intelligence**
