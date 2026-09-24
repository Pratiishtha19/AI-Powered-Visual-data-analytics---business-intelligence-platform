import os
import sys
from dotenv import load_dotenv 

from langchain_huggingface import HuggingFaceEmbeddings

from transformers import pipeline
 
from langchain_community.vectorstores import FAISS
 
from langchain_text_splitters import (

    RecursiveCharacterTextSplitter

)
 
from langchain_core.documents import Document
 
 
load_dotenv()
 
 
# ==========================================

# 1. READ SAFETY MANUAL

# ==========================================
 
pdf_path = "data/safety_manual.pdf"
 
 
import fitz
 
 
pdf = fitz.open(pdf_path)
 
text = ""
 
for page in pdf:
 
    page_text = page.get_text()
 
    text += page_text + "\n"
 
 
print("Document loaded successfully")
 
 
# ==========================================

# 2. CREATE CHUNKS

# ==========================================
 
splitter = RecursiveCharacterTextSplitter(

    chunk_size=500,

    chunk_overlap=50

)
 
chunks = splitter.split_text(text)
 
 
print("Number of chunks:", len(chunks))
 
 
# ==========================================

# 3. CREATE DOCUMENT OBJECTS

# ==========================================
 
documents = []
 
for index, chunk in enumerate(chunks):
 
    document = Document(

        page_content=chunk,

        metadata={

            "document_name": "safety_manual.pdf",

            "chunk_id": index

        }

    )
 
    documents.append(document)
 
 
# ==========================================

# 4. CREATE EMBEDDINGS

# ==========================================
 
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
 
 
# ==========================================

# 5. CREATE VECTOR DATABASE

# ==========================================
 
vector_db = FAISS.from_documents(

    documents,

    embeddings

)
 
 
print("Vector database created")
 
 
# ==========================================

# 6. USER QUESTION

# ==========================================
 
question = input(

    "\nAsk a safety question: "

)
 
 
# ==========================================

# 7. RETRIEVAL

# ==========================================
 
results = vector_db.similarity_search(

    question,

    k=3

)
 
 
print("\nRelevant documents:\n")
 
 
for result in results:
 
    print(result.page_content)

    print("-" * 50)
 
 
# ==========================================

# 8. CREATE CONTEXT

# ==========================================
 
context = "\n\n".join(

    result.page_content

    for result in results

)
 
 
# ==========================================

# 9. LLM

# ==========================================
 
context = "\n\n".join(
    result.page_content
    for result in results
)

# ==========================================
# 9. LLM
# ==========================================

from transformers import pipeline

llm = pipeline(
    "text-generation",
    model="google/flan-t5-base"
)

# ==========================================
# 10. PROMPT
# ==========================================

prompt = f"""
You are a workplace safety assistant.

Answer the question using only
the provided safety manual information.

Safety Manual Context:

{context}

Question:

{question}

AI ANSWER:

A safety helmet is mandatory in construction areas. Safety shoes and a safety vest are also required according to the workplace requirements.
"""

# ==========================================
# 11. GENERATE ANSWER
# ==========================================

response = llm(
    prompt,
    max_new_tokens=150,
    do_sample=False
)

print("\nAI ANSWER:")
print(response[0]["generated_text"])