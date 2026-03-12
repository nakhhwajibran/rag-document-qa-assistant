import os
from dotenv import load_dotenv, dotenv_values
import shutil

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores.utils import filter_complex_metadata

import nltk
import ssl
   
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")

load_dotenv()
config = dotenv_values(".env")
gptModel = config.get("GPT_MODEL")
os.environ["ANONYMIZED_TELEMETRY"] = "False"

working_dir = os.path.dirname(os.path.abspath((__file__)))
docs_dir = os.path.join(working_dir, 'docs')

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

llm = ChatOpenAI(
    model=gptModel,
    temperature=0
)

def process_document_to_chroma_db(file_name):
    # Load the PDF document using UnstructuredPDFLoader
    db_path = f"{working_dir}/doc_vector_db"

    loader = UnstructuredFileLoader(f"{docs_dir}/{file_name}",mode="elements")
    documents = loader.load()

    #Split the text into chunks ifro embedding

    text_split = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )

    texts = text_split.split_documents(documents)
    # Store the document chunks in a chroma vector database

    texts = filter_complex_metadata(texts)

    vectordb = Chroma(
        persist_directory=db_path,
        embedding_function=embedding
    )

    vectordb.add_documents(texts)
    vectordb.persist()

    return vectordb

def answer_question(user_question, vectordb):
    #load the persistent Chroma vector database
    retriever = vectordb.as_retriever(search_kwargs={"k":3})

    #Create a RetrievalQs chain to answer user question using OpenAi
    qa_chain = RetrievalQA.from_chain_type(
        llm =llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    response = qa_chain.invoke({"query":user_question})

    answer = response["result"]
    source_documents = response["source_documents"]

    return answer, source_documents