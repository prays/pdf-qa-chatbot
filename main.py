"""Python file to serve as the frontend"""
import streamlit as st
import openai
import pickle
import urllib.request
from streamlit_chat import message

from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyMuPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from openai.error import OpenAIError


import os

# Load using user OPENAI API Key
API_KEY = st.sidebar.text_input('Enter your API key', type="password")
os.environ["OPENAI_API_KEY"] = API_KEY



def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory {directory} created.")
    else:
        print(f"Directory {directory} already exists.")

# Clean file in data/ dir
def delete_files_in_directory(directory):
    if not os.path.isdir(directory):
        raise ValueError(f"{directory} is not a valid directory.")

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            print(f"{file_path} is a directory. Skipping...")

def download_pdf(url, output_path):
    urllib.request.urlretrieve(url, output_path)

# create_directory_if_not_exists("data")
# delete_files_in_directory("data/")

def clear_submit():
    st.session_state["submit"] = False


pdf_url = st.text_input('Insert pdf url link')
button_process = st.button("Process Data")



emb_path = "data/emb.pickle"
file_path = "data/file.pdf"

# @st.cache_data(allow_output_mutation=True)
def process_file(file_path, emb_path):
    print("Download pdf...")
    download_pdf(pdf_url, file_path)
    st.write(f"Saved pdf to {file_path}")

    # Construct loader
    print("Construct loader...")
    loader = PyMuPDFLoader(file_path)
    data = loader.load()

    # Split docs to chunk
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(data)

    # Convert to embedding
    print("Construct embeddings...")
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_documents(
                                documents, 
                                embeddings
                )

    with open(emb_path, "wb") as f: 
        pickle.dump(docsearch, f)

if button_process:
    if pdf_url:
        if API_KEY:
            process_file(file_path, emb_path)
            st.write("Done")
        else:
            st.error("Please insert your OpenAI API key!")
    else:
        st.warning("Please give pdf url link.")



# From here down is all the StreamLit UI.
st.header("PDF Chatbot Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    st.header("Ask me something about the pdf files:")
    input_text = st.text_area("You:", on_change=clear_submit)

    return input_text

user_input = get_text()

button = st.button("Submit")

if button:
    if API_KEY:
        if not pdf_url:
            st.error("Please insert pdf link!")
        elif not user_input:
            st.error("Please enter a question!")
        else:
            st.session_state["submit"] = True
            try:
                with open(emb_path, "rb") as f: 
                    docsearch = pickle.load(f)
                result_docs = docsearch.similarity_search(user_input)
                chain = load_qa_chain(ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"), chain_type="stuff")
                answer = chain({"input_documents": result_docs, "question": user_input}, return_only_outputs=True)
                st.session_state.past.append(user_input)
                st.session_state.generated.append(answer["output_text"])
            except OpenAIError as e:
                st.error(e._message)
            if st.session_state['generated']:
                for i in range(len(st.session_state['generated'])-1, -1, -1):
                    message(st.session_state["generated"][i], key=str(i))
                    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    else:
        st.error("Please insert your OpenAI API key!")

