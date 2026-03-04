# Swiggy Annual Report — RAG Question Answering System

> **Retrieval-Augmented Generation (RAG) using LangChain, Groq LLM (Llama 3), FAISS, and HuggingFace Embeddings**

---

## Table of Contents

- [Overview](#overview)
- [Objective](#objective)
- [Tech Stack](#tech-stack)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Pipeline Breakdown](#pipeline-breakdown)
- [Sample Questions to Ask](#sample-questions-to-ask)
- [Dataset Source](#dataset-source)
- [Conclusion](#conclusion)

---

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** based **Question Answering system** that allows users to interactively query the **Swiggy Annual Report** using natural language.

Instead of relying on general LLM knowledge (which can hallucinate), the system **grounds every answer strictly in the document content** — meaning it only answers based on what is written in the Swiggy Annual Report PDF.

The pipeline extracts text from the PDF, splits it into meaningful chunks, embeds those chunks using a sentence transformer, stores them in a FAISS vector database, and at query time retrieves the most relevant chunks and passes them to **Groq's Llama 3.3-70B** to generate a grounded, accurate answer.

---

## Objective

The goal of this assignment is to:

- Design and implement a production-style RAG pipeline using real-world business documents
- Enable users to ask questions in natural language about Swiggy's business, financials, and operations
- Ensure the system **does not hallucinate** — it must respond strictly based on the document
- Demonstrate strong understanding of document processing, vector stores, embeddings, and LLM chaining

---

## Tech Stack

| Component | Tool / Library |
|---|---|
| PDF Parsing | `PyMuPDF (fitz)` |
| Text Chunking | `LangChain RecursiveCharacterTextSplitter` |
| Embeddings | `HuggingFace - all-MiniLM-L6-v2` |
| Vector Database | `FAISS (Facebook AI Similarity Search)` |
| LLM | `Groq API — Llama 3.3 70B Versatile` |
| RAG Framework | `LangChain` |
| Interface | `CLI (Interactive Loop)` |
| Environment | `Google Colab` |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INDEXING PIPELINE                        │
│                                                                 │
│  Swiggy Annual Report PDF                                       │
│          │                                                      │
│          ▼                                                      │
│  ┌──────────────────┐                                           │
│  │  PyMuPDF Parser  │  ← Extracts text page-by-page             │
│  └──────────────────┘                                           │
│          │                                                      │
│          ▼                                                      │
│  ┌───────────────────────────┐                                  │
│  │  RecursiveCharacterSplitter│  ← chunk_size=800, overlap=150  │
│  └───────────────────────────┘                                  │
│          │                                                      │
│          ▼                                                      │
│  ┌──────────────────────────────┐                               │
│  │  HuggingFace Embeddings      │  ← all-MiniLM-L6-v2           │
│  │  (sentence-transformers)     │                               │
│  └──────────────────────────────┘                               │
│          │                                                      │
│          ▼                                                      │
│  ┌──────────────────┐                                           │
│  │  FAISS VectorDB  │  ← Saved locally as swiggy_vector_db      │
│  └──────────────────┘                                           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        QUERY PIPELINE                           │
│                                                                 │
│  User Query (Natural Language)                                  │
│          │                                                      │
│          ▼                                                      │
│  ┌──────────────────┐                                           │
│  │  FAISS Retriever │  ← Top-5 semantic similarity search       │
│  └──────────────────┘                                           │
│          │                                                      │
│          ▼                                                      │
│  ┌────────────────────────────┐                                 │
│  │  Format Retrieved Chunks   │  ← With page metadata           │
│  └────────────────────────────┘                                 │
│          │                                                      │
│          ▼                                                      │
│  ┌─────────────────────────────────────┐                        │
│  │  ChatPromptTemplate + Groq Llama 3  │  ← Grounded answer     │
│  └─────────────────────────────────────┘                        │
│          │                                                      │
│          ▼                                                      │
│  ┌──────────────────┐                                           │
│  │  StrOutputParser │  ← Clean string output                    │
│  └──────────────────┘                                           │
│          │                                                      │
│          ▼                                                      │
│        Answer printed to CLI                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
swiggy-rag/
│
├── swiggy_rag.py               # Main RAG pipeline script
├── README.md                   # This file
└── requirements.txt            # Python dependencies (optional)
```

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- Google Colab (recommended) or local environment
- A **Groq API Key** — get one free at [https://console.groq.com](https://console.groq.com)

### Install Dependencies

Run the following in your Colab notebook or terminal:

```bash
pip install langchain langchain-core langchain-community langchain-huggingface \
            langchain-groq langchain-text-splitters sentence-transformers \
            faiss-cpu pypdf pillow pytesseract pymupdf
```

### Set Up API Key

When prompted during runtime, enter your Groq API key:

```
Enter your GROQ API key: ••••••••••••••••
```

---

## How to Run

### On Google Colab

1. Open the notebook `swiggy_rag.ipynb` in Google Colab
2. Run all cells from top to bottom
3. When prompted, **upload the Swiggy Annual Report PDF**
4. Enter your **Groq API Key**
5. Once the pipeline is ready, start asking questions in the CLI loop:

```
Ask a question about Swiggy Annual Report: What is Swiggy's total revenue?

Answer:

According to the Swiggy Annual Report [Page 42], Swiggy's total revenue from operations...
```

6. Type `exit` to quit the question-answering loop

### On Local Machine

1. Place the Swiggy Annual Report PDF in the same directory
2. Update `PDF_PATH` in the script to point to your local file:
   ```python
   PDF_PATH = "Swiggy_Annual_Report_2024.pdf"
   ```
3. Remove the `from google.colab import files` block (lines used for Colab file upload)
4. Run the script:
   ```bash
   python swiggy_rag.py
   ```

---

## Pipeline Breakdown

### 1. Document Processing

- The PDF is opened using **PyMuPDF (`fitz`)**, which extracts clean text page-by-page
- Whitespace and formatting noise is cleaned using regex: `re.sub(r'\s+', ' ', text).strip()`
- Each page is stored as a `LangChain Document` object with metadata including `page number`, `type`, and `source`

### 2. Smart Text Chunking

- `RecursiveCharacterTextSplitter` splits documents into chunks of **800 characters** with **150 character overlap**
- Overlap ensures that sentences spanning chunk boundaries are not lost, preserving semantic continuity
- This results in hundreds of fine-grained, overlapping chunks ready for embedding

### 3. Embeddings

- Each chunk is converted into a **dense vector representation** using `sentence-transformers/all-MiniLM-L6-v2` from HuggingFace
- This is a lightweight, high-quality model optimized for **semantic similarity tasks**
- Embeddings capture the meaning of each chunk, not just keywords

### 4. Vector Database (FAISS)

- All chunk embeddings are stored in a **FAISS index** (Facebook AI Similarity Search)
- FAISS enables fast, scalable **approximate nearest neighbor (ANN) search**
- The index is saved locally as `swiggy_vector_db/` for reuse without re-indexing

### 5. Retrieval

- The retriever performs **semantic similarity search** over the FAISS index
- For every user query, it retrieves the **top 5 most relevant chunks** from the document
- Retrieved chunks include page metadata, displayed as `[Page X]` in the context

### 6. LLM — Groq Llama 3.3 70B

- The retrieved chunks and the user's question are formatted into a **structured prompt**
- The prompt strictly instructs the LLM to answer **only from the provided context**
- If the answer isn't found in the document, the model responds: *"I couldn't find this in the Swiggy Annual Report."*
- **Groq's Llama 3.3-70B** is used for its speed, accuracy, and free API tier

### 7. Question Answering Interface

- A simple **CLI loop** lets users ask unlimited questions until they type `exit`
- The final answer is printed cleanly to the terminal

---

## Sample Questions to Ask

Here are some great questions to test the system with:

```
What is Swiggy's total revenue for the financial year?
How many orders did Swiggy fulfill in FY2024?
What are the key business segments of Swiggy?
```

---

## Dataset Source

- **Document:** Swiggy Annual Report (FY 2023-24)
- **Source:** [https://www.swiggy.com/corporate/](https://www.swiggy.com/corporate/)
- **Format:** PDF
- **Availability:** Publicly available on Swiggy's official Investor Relations page

---

## Conclusion

This project successfully demonstrates a complete, end-to-end **RAG-based Question Answering pipeline** built on a real-world financial document. The system:

- **Accurately retrieves** the most relevant sections of the Swiggy Annual Report for any given query
- **Generates grounded answers** using a state-of-the-art LLM (Llama 3.3 70B via Groq) with zero hallucination by design
- **Covers all functional requirements** — document processing, embedding, vector storage, retrieval, generation, and a working QA interface
- Is **modular and extensible** — each component (embeddings, vector DB, LLM) can be swapped independently

The RAG architecture ensures that the system is both **reliable** (it won't invent facts) and **flexible** (it can handle a wide variety of natural language questions about the report). This project lays a strong foundation for building production-grade document intelligence applications.

---

---
