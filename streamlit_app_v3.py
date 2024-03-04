import streamlit as st
import requests
import pandas as pd


from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI



st.title('Chat with CSV')
openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password', key="api_key")


def csv_filepath(file):
    if file is not None:
        # Save the uploaded file locally
        with open("uploaded_csv.csv", "wb") as f:
            f.write(file.read())
        file_path = "uploaded_csv.csv"
    else:
        file_path = None

    return file_path
    
def generate_response(uploaded,question_text,openai_api_key):

    df= pd.read_csv(csv_filepath(uploaded))
    df.head()
    
    engine = create_engine("sqlite:///uploaded.db")
    df.to_sql("titanic", engine, index=False, if_exists='replace')

    db = SQLDatabase(engine=engine)
    print(db.dialect)
    print(db.get_usable_table_names())
    

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0,openai_api_key=openai_api_key )
    agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

    response = agent_executor.invoke({"input": question_text})

    st.info(response.get("output"))



# Main Streamlit app
with st.form('my_form'):
    # File upload section
    uploaded = st.file_uploader("Choose a CSV file")
    submitted_upload = st.form_submit_button('Upload')

    question_text = ""
    submitted_text = False
    if not openai_api_key.startswith('sk-'):
            st.warning('Please enter a valid OpenAI API key!', icon='⚠')
            
    # If a file is uploaded
    if uploaded is not None and submitted_upload:
        # Text input section
        question_text = st.text_area('Enter text:', 'Type, what do you want to know from this document?')
        #submitted_text = st.form_submit_button('Submit')

        # Check for missing or invalid inputs
        if not question_text:
            st.warning('Please enter your question!', icon='⚠')
            
        
    # Warning for missing PDF file
    if not uploaded and submitted_upload:
        st.warning('Please upload a CSV file!', icon='⚠')


    # Call the function to generate response if all conditions are met
    if uploaded and question_text and openai_api_key.startswith('sk-') :
        
        try:
            generate_response(uploaded, question_text, openai_api_key)
        except requests.exceptions.RequestException as e:
            st.error(f"OpenAI API Error: {e}")
