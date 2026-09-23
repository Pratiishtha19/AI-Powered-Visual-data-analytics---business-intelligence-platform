# 🤖 AI-Powered Visual Data Analytics and Business Intelligence

An AI-powered visual analytics system that combines **computer vision, PPE detection, document processing, vector databases, and Retrieval-Augmented Generation (RAG)** to build an intelligent safety knowledge and monitoring pipeline.

The project is designed as a practice implementation for **Milestone 3**, integrating the knowledge gained from document processing, embeddings, vector search, LLM-based question answering, and YOLO-based computer vision.

---

## 📌 Project Overview

The system combines two major capabilities:

### 📄 1. Document Intelligence & RAG

A safety manual PDF is processed through a complete document-processing pipeline:

**PDF → Text Extraction → Cleaning → Chunking → Embeddings → ChromaDB → Semantic Search → LLM Answer**

The system converts the safety manual into searchable knowledge chunks and stores their vector representations in **ChromaDB**.

When a user asks a question, the system searches the knowledge base for relevant information and provides the retrieved context to an LLM for generating an answer.

### 🦺 2. Computer Vision & PPE Detection

The project also includes YOLO-based computer vision components for detecting objects and Personal Protective Equipment (PPE).

The PPE detection pipeline uses a fine-tuned YOLO model (`best.pt`) to identify safety equipment such as helmets, vests, or other trained PPE classes.

---

# ✨ Key Features

* 📄 PDF / DOCX / TXT document loading
* 🧹 Text cleaning and normalization
* ✂️ Intelligent text chunking
* 🏷️ Document and chunk metadata extraction
* 🧠 Embedding-based semantic search
* 🗄️ ChromaDB vector knowledge base
* 🔎 Retrieval-Augmented Generation (RAG) workflow
* 🤖 YOLO object detection
* 🦺 Fine-tuned PPE detection
* 📷 Image-based detection
* 🎥 Video-based detection
* 📹 Real-time webcam monitoring
* 🧩 Modular Python architecture

---

# 🏗️ Project Structure

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
│
├── ppe_model.py
├── vision.py
├── yolo_model.py
│
├── milestone3_practice.py
├── requirements.txt
└── README.md
```

---

# 🔄 System Architecture

## 📚 RAG Knowledge Pipeline

```text
                 safety_manual.pdf
                        │
                        ▼
                ┌─────────────────┐
                │ DocumentLoader  │
                └────────┬────────┘
                         │
                         ▼
                   Raw Text
                         │
                         ▼
                ┌─────────────────┐
                │  TextCleaner    │
                └────────┬────────┘
                         │
                         ▼
                  Cleaned Text
                         │
                         ▼
                ┌─────────────────┐
                │     Chunker      │
                └────────┬────────┘
                         │
                         ▼
                  Text Chunks
                         │
                         ▼
                ┌─────────────────┐
                │   Embeddings    │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │    ChromaDB     │
                │ Vector Database  │
                └────────┬────────┘
                         │
                    User Query
                         │
                         ▼
                Semantic Retrieval
                         │
                         ▼
                Relevant Chunks
                         │
                         ▼
                       LLM
                         │
                         ▼
                  Final Answer
```

---

# 🦺 Computer Vision Pipeline

```text
       Image / Video / Webcam
                │
                ▼
        ┌─────────────────┐
        │   YOLO Model    │
        └────────┬────────┘
                 │
                 ▼
        Object / PPE Detection
                 │
                 ▼
       Bounding Boxes + Confidence
                 │
                 ▼
          Detection Results
```

---

# ⚙️ Technologies Used

| Technology               | Purpose                            |
| ------------------------ | ---------------------------------- |
| Python                   | Core programming language          |
| YOLOv8                   | Object and PPE detection           |
| Ultralytics              | YOLO implementation                |
| ChromaDB                 | Vector database                    |
| Sentence Transformers    | Text embeddings                    |
| PyMuPDF / PDF processing | PDF text extraction                |
| NumPy                    | Numerical processing               |
| OpenCV                   | Image/video processing             |
| LLM                      | Natural-language answer generation |
| Git & GitHub             | Version control                    |

---

# 📁 Module Description

## `src/documents/document_loader.py`

Responsible for loading supported documents such as:

* PDF
* DOCX
* TXT

### Main class

```text
DocumentLoader
```

### Main method

```python
.load(path)
```

---

## `src/documents/text_cleaner.py`

Cleans and normalizes extracted document text.

Typical processing includes:

* Removing unnecessary whitespace
* Normalizing text
* Removing unwanted characters
* Preparing text for chunking

### Main class

```text
TextCleaner
```

### Main method

```python
.clean(text)
```

---

## `src/documents/chunker.py`

Splits large documents into smaller chunks suitable for embedding and retrieval.

### Main class

```text
Chunker
```

### Main method

```python
.split(text)
```

Chunking makes it easier for the vector database to retrieve only the relevant parts of a document.

---

## `src/documents/metadata.py`

Extracts useful information related to documents and chunks.

Examples include:

* File information
* Document source
* Chunk information
* Text-related metadata

### Main class

```text
MetadataExtractor
```

### Main method

```python
.extract(path, text)
```

---

## `src/knowledge_base/database.py`

Handles the vector knowledge base using **ChromaDB**.

It is responsible for storing document chunks and searching for relevant informatio
