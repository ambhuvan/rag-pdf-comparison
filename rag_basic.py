import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Load PDF
loader = PyPDFLoader("sample.pdf")  # Place your PDF in same folder
docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Embed and store in vector DB
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # safe default model
vectorstore = Chroma.from_documents(chunks, embeddings)

# Create retrieval-based QA chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=vectorstore.as_retriever()
)

# Ask question
query = "What is the document about?"
response = qa.run(query)
print(response)
