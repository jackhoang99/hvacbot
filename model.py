import streamlit as st
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Replicate
import replicate



# Setting up the environment variable
REPLICATE_API_TOKEN = st.secrets["REPLICATE_API_TOKEN"]

# Define constants
DB_FAISS_PATH = 'vectorstore/db_faiss'
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
modelrp=st.secrets['model']

def load_llm():
    return Replicate(
        model=modelrp,
        input={"temperature": 0.6, "max_length": 512, "top_p": 1},
    )

modelhf=st.secrets["model"]

def load_qa_bot():
    # Initialize loaders, embeddings, and other components as necessary
    # Return the initialized RetrievalQA object
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    
    return RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 1}),
                                       return_source_documents=False,
                                       chain_type_kwargs={'prompt': prompt})


def app():
    st.title('Airlast\'s HVAC Q&A Bot')
    user_input = st.text_area("Ask anything related to HVAC:")
    if st.button('Submit'):
        if user_input:
            with st.spinner('Loading and processing...'):
                try:
                    qa_bot = load_qa_bot()
                    response = qa_bot({'query': user_input})
                    st.write(response['result'])
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}") 
                    



if __name__ == "__main__":
    app()
