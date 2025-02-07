Custom Chatbot using LangChain

This project involves creating a custom chatbot powered by LangChain, capable of extracting data from a specified website, storing embeddings in a vector store, and serving the chatbot's functionality through a Flask RESTful API. The chatbot leverages the LangChain library for data extraction and OpenAI embeddings for efficient similarity search.
Features

    Data Extraction:
        Scrapes data from the Brainlox Technical Courses webpage using LangChain's WebBaseLoader.

    Embeddings Creation:
        Processes the extracted text and generates numerical embeddings using OpenAI models.
        Stores these embeddings in a FAISS vector store for efficient retrieval.

    Flask RESTful API:
        A Flask-based REST API allows users to query the chatbot for relevant information.
        Handles user queries by searching the vector store and returning the best-matched responses.

Requirements

To run this project, you need:

    Python 3.8 or higher
    Required Python libraries:
        langchain
        openai
        HUGGINGFACE 
        faiss-cpu
        flask
        flask-restful
        beautifulsoup4
        requests

You can install all dependencies using:

pip install langchain openai faiss-cpu flask flask-restful beautifulsoup4 requests

Extract Data from the Website: Use the WebBaseLoader from LangChain to scrape data from the website:

from langchain.document_loaders import WebBaseLoader

url = "https://brainlox.com/courses/category/technical"
loader = WebBaseLoader(url)
documents = loader.load()

Create Embeddings: Process the extracted text, split it into smaller chunks, and generate embeddings using OpenAI's/HuggingFace embedding model:

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(texts, embedding_model)
vectorstore.save_local("faiss_index")

Run the Flask API: Start the Flask RESTful API to handle chatbot conversations:

python app.py

Test the Chatbot: Use Postman, cURL, or any HTTP client to send POST requests to the API. Example:

    curl -X POST http://127.0.0.1:5000/chat \
    -H "Content-Type: application/json" \
    -d '{"query": "What courses are available?"}'

Folder Structure

project/
│
├── app.py                  # Flask RESTful API
├── embeddings/             # Folder to store vector embeddings (FAISS index)
├── requirements.txt        # List of dependencies
├── README.md               # Project documentation (this file)
└── scripts/                # Python scripts for data extraction and processing

Usage

This project can be used for:

    Extracting structured data from websites.
    Creating and storing embeddings for text-based similarity searches.
    Building chatbots capable of interacting with users through a REST API.
