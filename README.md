# Multi PDF Search with Streamlit and Google Generative AI

This project is a Streamlit application that allows users to upload multiple PDF files, process the text, and interact with the content through a question-answering interface powered by Google Generative AI and FAISS vector store.

## Features

- Upload and process multiple PDF files.
- Extract text from the uploaded PDFs.
- Chunk the extracted text for efficient processing.
- Convert text chunks into vectors using Google Generative AI embeddings.
- Perform similarity search on the vectorized chunks.
- Answer questions based on the content of the PDFs using a conversational AI model.

## Requirements

- Python 3.7+
- Streamlit
- PyPDF2
- langchain
- langchain_google_genai
- google-generativeai
- faiss-cpu

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/multi-pdf-search.git
    cd multi-pdf-search
    ```

2. **Create and activate a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Setup

1. **Set your Google API Key:**

    Replace the placeholder with your actual Google API Key in the script.

    ```python
    os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"
    ```

2. **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

## Usage

1. Open the application in your web browser (usually at `http://localhost:8501`).
2. Use the sidebar to upload multiple PDF files.
3. Click on "Submit and process" to extract and vectorize the text from the PDFs.
4. Once processing is complete, enter your question in the text input box.
5. The application will provide an answer based on the content of the uploaded PDFs.

## Code Overview

### Main Functions

- `get_pdf_text(pdf_docs)`: Extracts text from the uploaded PDF documents.
- `get_text_chunks(text)`: Splits the extracted text into manageable chunks.
- `get_vector_store(text_chunks)`: Converts the text chunks into vectors and saves them locally.
- `get_conversational_chain()`: Creates a conversational AI chain for answering questions.
- `user_input(user_question)`: Handles the user input, performs similarity search, and retrieves the response.
- `main()`: Initializes the Streamlit application and handles the user interface.

### Example Code Snippet

```python
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Define functions here...

def main(): 
    st.set_page_config("Multi PDF Search")
    st.header("Chat with PDFs using Generative AI")

    user_questions = st.text_input("Question")

    if user_questions:
        user_input(user_questions)
    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF", accept_multiple_files=True)
        if st.button('Submit and process'):
            with st.spinner("Uploading and Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()

