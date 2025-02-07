#!/usr/bin/env python
# coding: utf-8

# In[31]:


import re
from langchain_community.document_loaders import WebBaseLoader

# Load data from website
url = "https://brainlox.com/courses/category/technical"
loader = WebBaseLoader(url)

# Extracted documents
documents = loader.load()

# Combine text from all extracted documents
raw_text = " ".join([doc.page_content for doc in documents])

# Regular expression to extract course details
pattern = re.findall(r'\$(\d+)per sessionLEARN (.*?)\s+.*?(\d+) LessonsView Details', raw_text)

# Formatting extracted data
courses = []
for price, name, lessons in pattern:
    courses.append(f"Course: {name}\nPrice: ${price} per session\nLessons: {lessons}\n")

# Print structured courses
cleaned_courses = "\n".join(courses)
print(cleaned_courses)


# In[32]:


from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# Load SentenceTransformer model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Convert course data to a format suitable for vector storage
documents = [
    f"Course: {name}, Price: ${price} per session, Lessons: {lessons}"
    for price, name, lessons in pattern
]

# Generate embeddings
vectors = [embedding_model.embed_query(doc) for doc in documents]

# Store in ChromaDB
vector_db = Chroma.from_texts(texts=documents, embedding=embedding_model, persist_directory="./chroma_db")

# Save the database
vector_db.persist()

print("✅ Course embeddings stored in ChromaDB successfully!")


# In[33]:


from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Load the stored embeddings
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

# Query function
def query_courses(user_query, top_k=3):
    results = db.similarity_search(user_query, k=top_k)
    return [doc.page_content for doc in results]

# Test Query
query = "Python programming basics"
retrieved_courses = query_courses(query)
for i, course in enumerate(retrieved_courses, 1):
    print(f"{i}. {course}")


# In[34]:


from flask import Flask
import threading

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, Flask is running inside Jupyter Notebook!"

# Function to run Flask in a separate thread
def run_flask():
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)

# Start Flask in a background thread
flask_thread = threading.Thread(target=run_flask)
flask_thread.start()


# In[35]:


import requests

url = "http://192.168.235.74:5001/"
response = requests.get(url)
print(response.text)


# In[37]:


from flask import Flask, request, jsonify
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

app = Flask(__name__)

# Load the stored embeddings
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

@app.route('/search', methods=['GET'])
def search_courses():
    user_query = request.args.get("query")
    if not user_query:
        return jsonify({"error": "Query parameter is required"}), 400
    
    results = db.similarity_search(user_query, k=3)
    courses = [doc.page_content for doc in results]

    return jsonify({"query": user_query, "results": courses})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)


# In[41]:


from flask import Flask, request, jsonify
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import threading

app = Flask(__name__)

# Load stored embeddings
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

@app.route('/search', methods=['GET'])
def search_courses():
    user_query = request.args.get("query")
    if not user_query:
        return jsonify({"error": "Query parameter is required"}), 400
    
    results = db.similarity_search(user_query, k=3)
    courses = [doc.page_content for doc in results]

    return jsonify({"query": user_query, "results": courses})

# Function to run Flask in a background thread
def run_flask():
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)

# Start Flask in a separate thread
flask_thread = threading.Thread(target=run_flask)
flask_thread.start()


# In[1]:


from flask import Flask, request, jsonify
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

app = Flask(__name__)

# Load the stored embeddings
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

# Query function
def query_courses(user_query, top_k=3):
    results = db.similarity_search(user_query, k=top_k)
    return [doc.page_content for doc in results]

# Define the search endpoint
@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("query")
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
    
    results = query_courses(query)
    return jsonify({"results": results})

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)


# In[ ]:


if __name__ == "__main__":
    from werkzeug.serving import run_simple
    run_simple("0.0.0.0", 5001, app, use_reloader=False, use_debugger=True)


# In[ ]:


import requests
response = requests.get("http://127.0.0.1:5001/search", params={"query": "Python programming"})
print(response.json())


# In[ ]:




