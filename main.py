from pinecone import Pinecone
import google.generativeai as genai

from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import Settings

from llama_index.readers.web import BeautifulSoupWebReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.pinecone import PineconeVectorStore

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from dotenv import load_dotenv
import os

from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin 
import re 
from llama_index.core import Document

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get('PINECONE_ENVIRONMENT')

llm = Gemini(api_key=GOOGLE_API_KEY)
embed_model = GeminiEmbedding(model_name="models/embedding-001")

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 1024


try:
    import PyPDF2
except ImportError:
    print("PyPDF2 not installed")

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyPDF2."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
            return text
    except Exception as e:
        print(f"Error extracting PDF with PyPDF2: {e}. Trying pdfminer.six")
        return extract_text_from_pdf_miner(pdf_path)

def extract_text_from_pdf_miner(pdf_path):
    """Extracts text from a PDF file using pdfminer.six."""
    try:
        text = extract_text(pdf_path)
        return text
    except Exception as e:
        print(f"Error extracting PDF with pdfminer.six:{e}")
        return ""

def clean_text(text):
    """Cleans up extracted text."""
    text=re.sub(r'\s+',' ',text) 
    text=text.replace('\n',' ')  
    text=text.strip()  
    return text


def scrape_website_with_pdfs(url):
    """
    Scraping a website, extracts text from HTML and PDFs,which returns a list of
    llama_index.core.Document objects which help make us embeddings.
    """
    documents=[]  
    try:
        response=requests.get(url)
        response.raise_for_status()
        soup=BeautifulSoup(response.content,'html.parser')

        html_text=soup.get_text(separator='\n',strip=True)
        html_text=clean_text(html_text)
        documents.append(Document(text=html_text, metadata={"source": "HTML", "url": url}))

        for link in soup.find_all('a', href=True):
            href = link['href']
            abs_url = urljoin(url, href)

            if abs_url.lower().endswith('.pdf'):
                print(f"Found PDF: {abs_url}")

                try:
                    pdf_response = requests.get(abs_url)
                    pdf_response.raise_for_status()

                    pdf_filename = "temp.pdf"
                    with open(pdf_filename, 'wb') as f:
                        f.write(pdf_response.content)

                    pdf_text=extract_text_from_pdf(pdf_filename)
                    pdf_text=clean_text(pdf_text)
                    documents.append(Document(text=pdf_text, metadata={"source": "PDF", "url": abs_url}))
                    os.remove(pdf_filename)

                except requests.exceptions.RequestException as e:
                    print(f"Error downloading PDF {abs_url}:{e}")
                except Exception as e:
                    print(f"Error processing PDF {abs_url}:{e}")

        return documents

    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}:{e}")
        return []


DATA_URL="https://iiitn.ac.in/"

try:
    pinecone_client=Pinecone(api_key=PINECONE_API_KEY,environment=PINECONE_ENVIRONMENT)
    print("Pinecone connection successful!")
    print(pinecone_client.list_indexes())
except Exception as e:
    print(f"Pinecone connection error: {e}")

# for index in pinecone_client.list_indexes():
#     print(index['name'])

index_description=pinecone_client.describe_index("medium-blog-text-embedding")
# print(index_description)

# FinalDocuments = scrape_website_with_pdfs(DATA_URL)
# print(FinalDocuments)

pinecone_index=pinecone_client.Index("medium-blog-text-embedding")
vector_store=PineconeVectorStore(pinecone_index=pinecone_index)

pipeline=IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=1024,chunk_overlap=20),
        embed_model
    ],
    vector_store=vector_store
)

# pipeline.run(documents=FinalDocuments)

index=VectorStoreIndex.from_vector_store(vector_store=vector_store)
retriever=VectorIndexRetriever(index=index,similarity_top_k=25)
query_engine=RetrieverQueryEngine(retriever=retriever)

responce=query_engine.query("What is the last date to pay hostel fees for 2025 year semester ?")
print(responce)
# responce : "The last date for hostel fee payment is December 26th, 2024, until 5 PM."
