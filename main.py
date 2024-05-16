from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from fastapi import FastAPI,Form,File,UploadFile
from typing import Annotated
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import chromadb
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_nomic import NomicEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings


# client_name=chromadb.HttpClient(host="http://127.0.0.1:8000",port=8000)
load_dotenv()

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

def loader_pdf(path):
    loader=PyPDFLoader(path)
    docs= loader.load()
    return docs

def splitter_pdf(docs):
    text_splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000,chunk_overlap=100)
    chunks= text_splitter.split_documents(docs)
    return chunks

def preprocess_data(pdf):
    docs= loader_pdf(pdf)
    chunks= splitter_pdf(docs)
    return chunks


def semantic_store(chunks,question):
    try:
        #embeddings= GoogleGenerativeAIEmbeddings(model="models/embedding-001",task_type="retrieval_query")
        embeddings= GPT4AllEmbeddings()
        vector_db=Chroma.from_documents(chunks,embeddings,collection_name="rag-chroma")
        relev_docs=vector_db.similarity_search(query=question)
        return relev_docs
    except ValueError as e:
        raise e
        


def generate_rag(rlv_docs,question):
    try:
        print("RAG PROMP1T")
        rag_prompt=PromptTemplate(template="""
        You are a helpful assistant, you primary task is provide feedback to user's question using the context provided bellow:\n
        These are the context and the user's question:
        -QUESTION: {question}
        -CONTEXT:{context}
                                
        IMPORTANT: Take your time to analyze the context , if the context doesn't contain any keyword related with the user's question , then respond telling that you don't have the necessary information to respond the question, nothing else.
    """,
        input_variables=["question","context"])
        llm=ChatGroq(model="llama3-8b-8192",temperature=0.1)
        output_parser = StrOutputParser()
        chain_rag=rag_prompt|llm|output_parser
        response= chain_rag.invoke({"question":question,"context":rlv_docs})
        return response
    except ValueError as e:
        raise e


@app.post("/generate")
async def generate(pdf:UploadFile,question:Annotated[str,Form()]):
    content_pdf=await pdf.read()
    file_path="test.pdf"
    with open(file_path,"wb") as f:
        f.write(content_pdf)
      

    print("PDF PATH")
    ## if file was created generate RAG
    if os.path.exists(file_path):
        try:
           ##TODO preprocess the file
           chunks=preprocess_data(file_path)
           ##TODO Obatin vector store
           relev_docs= semantic_store(chunks=chunks,question=question)
           ##TODO build RAG CHAIN 
           response= generate_rag(rlv_docs=relev_docs,question=question)
           return  {response}
        except ValueError as e:
           return {"ERROR":e}
    else:
        return {"Error":"There was a problem uploading the file"}

     


    