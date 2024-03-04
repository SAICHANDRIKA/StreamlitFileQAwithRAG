import streamlit as st
import requests

from langchain_community.llms import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain import hub

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel


st.title('Chat with PDF')
openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password', key="api_key")


def pdf_load(file):
    if file is not None:
        # Save the uploaded file locally
        with open("uploaded_pdf.pdf", "wb") as f:
            f.write(file.read())
        file_path = "data/uploaded_pdf.pdf"
    else:
        file_path = None

    loader = PyPDFLoader(file_path)
    return loader.load()

def pdf_docs_split(fdocs):
    # Splitting docs into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(fdocs)

def pdf_embedding_store(texts,openai_api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    store = Chroma.from_documents(texts,embeddings)
    return store

def format_docs(docs):
    return "/n/n".join(doc.page_content for doc in docs)
    
def generate_response(uploaded,question_text,openai_api_key):
    #Indexing - Load
    docs = pdf_load(uploaded)

    #Indexing - Split
    texts = pdf_docs_split(docs)

    #Indexing - Store
    db = pdf_embedding_store(texts,openai_api_key)

    #Retrieve 
    retriever = db.as_retriever(search_type="similarity",search_kwargs={"k":3})

    #Generate
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,openai_api_key=openai_api_key)
    prompt = hub.pull("rlm/rag-prompt")

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))   | prompt | llm | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)
   
    
    st.text(" ".join(str(chunk.get('answer', '')) for chunk in rag_chain_with_source.stream(question_text)))




# Main Streamlit app
with st.form('my_form'):
    # File upload section
    uploaded = st.file_uploader("Choose a PDF file")
    
    submitted_upload = st.form_submit_button('Upload')

    question_text = ""
    submitted_text = False
    if not openai_api_key.startswith('sk-'):
            st.warning('Please enter a valid OpenAI API key!', icon='⚠')
            
    # If a file is uploaded
    if uploaded is not None and submitted_upload:

        # Text input section
        question_text = st.text_area('Enter text:', 'Type, what do you want to search in this document?')
        #submitted_text = st.form_submit_button('Submit')

        # Check for missing or invalid inputs
        if not question_text:
            st.warning('Please enter your question!', icon='⚠')
            
        
    # Warning for missing PDF file
    if not uploaded and submitted_upload:
        st.warning('Please upload a PDF file!', icon='⚠')


    # Call the function to generate response if all conditions are met
    if uploaded and question_text and openai_api_key.startswith('sk-') :
        
        try:
            generate_response(uploaded, question_text, openai_api_key)
        except requests.exceptions.RequestException as e:
            st.error(f"OpenAI API Error: {e}")
