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

os.environ["GOOGLE_API_KEY"] = "AIzaSyDLG4C5hY9JSvZk1pkkVrxOWZlgVzbKFtk"
genai.configure(api_key =os.environ["GOOGLE_API_KEY"] )

#Pdf Collection

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

#Text Chuck creation

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    chunk=text_splitter.split_text(text)
    return chunk

# converting the chunks to vectore
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model ="models/embedding-001")
    vectore_store = FAISS.from_texts(text_chunks,embedding = embeddings)
    vectore_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context","questions"])
    chain = load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain


    #user input
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model ="models/embedding-001")
    new_db = FAISS.load_local("faiss_index",embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    #response = chain(
        #{"input_documents":docs,"questions":user_question}
        #, return_only_ouputs=True)
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    
    print(response)

    st.write("reply: ",response['output_text'])


#Streamlit initilaizing

def main(): 
    st.set_page_config("Multi PDF Search")
    st.header("Chat with | PDF using Gemeini👌")

    user_questions = st.text_input("Question")

    if user_questions:
        user_input(user_questions)
    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF",accept_multiple_files=True)
        if st.button('Submit and process'):
            with st.spinner("Uploading and Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done 🔥")
if __name__ == "__main__":
    main()