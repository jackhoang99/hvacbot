import streamlit as st
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Replicate
import replicate
import time



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
modelrp=st.secrets['modelrp']

def load_llm():
    return Replicate(
        model=modelrp,
        model_kwargs={"temperature": 0.5, "max_length": 512, "top_p": 1},
    )

modelhf=st.secrets["modelhf"]

def load_qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name=modelhf,
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    
    return RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt})


def app():
    st.title('Airlast\'s HVAC Q&A Bot')
    user_input = st.text_area("Ask anything related to HVAC:")
    if st.button('Submit'):
        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0)
        progress_caption = st.caption(progress_text)
        
        try:
            my_bar.progress(10)

            qa_bot = load_qa_bot()
            my_bar.progress(50)

            response = qa_bot({'query': user_input})
            my_bar.progress(100)

            st.write(response['result'])

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

        progress_caption.empty()
        my_bar.empty()
                    



if __name__ == "__main__":
    app()