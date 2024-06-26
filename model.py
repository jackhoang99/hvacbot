import replicate
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Replicate
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Setting up the environment variable
REPLICATE_API_TOKEN = st.secrets["REPLICATE_API_TOKEN"]
# Define constants
DB_FAISS_PATH = 'vectorstore/db_faiss'
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Don't include chapter or figure in your final answer. Do not acknowledge my request with "sure" or in any other way besides going straight to the answer. 
Don't include base on information provided in your final answer.
Don't mention base on information provided in your final answer.
Context: {context}
Question: {question}
Helpful answer:
"""
modelrp="meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"
def load_llm():
    return Replicate(
        model=modelrp,
        model_kwargs={"temperature": 0.5, "max_new_tokens": 500 , "top_p": 1},
    )

def load_qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings,allow_dangerous_deserialization=True)
    llm = load_llm()
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    
    return RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt})

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
placeholder = st.empty()
actual_email = st.secrets["actual_email"]
actual_password = st.secrets["actual_password"]
# Display login form if user is not logged in
if not st.session_state.logged_in:
    with placeholder.form("login"):
        st.markdown("#### Enter your credentials")
        email = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        if submit and email != actual_email and password != actual_password:
            st.error("Incorrect username or password")
        if submit and email == actual_email and password == actual_password:
            st.session_state.logged_in = True
            placeholder.empty()
# Display bot if user is logged in
if st.session_state.logged_in:
    st.title('Airlast\'s HVAC Q&A Bot')
    user_input = st.text_area("Ask anything related to HVAC:")
    
    if st.button('Submit'):
        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0)
        progress_caption = st.caption(progress_text)
        
        try:
            my_bar.progress(10)
            qa_bot = load_qa_bot()
            my_bar.progress(65)
            response = qa_bot({'query': user_input})
            my_bar.progress(100)
            st.write(response['result'])
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        progress_caption.empty()
        my_bar.empty()