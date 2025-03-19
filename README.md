# RAG-Based College Website Information Retriever

This project implements a **Retrieval-Augmented Generation (RAG)** system that extracts, indexes, and makes searchable content from IIIT Nagpur's website, including both HTML content and PDF documents. The system allows users to query institutional information through natural language and receive accurate responses based on the indexed content.

## Why It's Unique

- **Simplified Information Access**: Eliminates the tedious process of manually searching through numerous PDFs on institutional websites.
- **Semantic Understanding**: Delivers targeted information by understanding the meaning behind user queries, not just keyword matching.
- **Time-Efficient**: Significantly reduces time spent locating important information like internship announcements, academic calendars, and policies.
- **Unified Knowledge Base**: Transforms unstructured institutional data scattered across multiple documents into an easily queryable resource.
- **Single Access Point**: Provides one interface to query information that would otherwise require browsing multiple web pages and PDF documents.

## Technical Architecture

- **LLM Integration**: Uses Google's **Gemini** model for embeddings and query expansion.
- **Vector Database**: Stores and retrieves document embeddings using **Pinecone**.
- **Document Processing**: Implements **LlamaIndex** for document parsing, chunking, and retrieval optimization.
- **Web Scraping**: Custom functionality to process both HTML content and PDF documents.
- **Query Enhancement**: Expands user queries for better retrieval results.

## Prerequisites

- Python 3.7+
- Google API key (for Gemini)
- Pinecone API key and environment
- Required Python packages (listed in `requirements.txt`)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Vedant988/WebPDF-RAG.git
cd WebPDF-RAG
```

### 2. Install dependencies
Install the required Python packages listed in requirements.txt:

```bash
pip install -r requirements.txt
```

### 3. Setup Environment Variables
Create a .env file in the root directory of the project and add your API keys:

```bash
GOOGLE_API_KEY=your-google-api-key
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=your-pinecone-environment
```


### 4. Run the Project
To start the information retrieval system, run the following command:
```bash
python main.py
```

## Thank You ...
