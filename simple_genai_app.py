import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Simple GenAI App"
os.environ["LANGSMITH_TRACING_V2"] = "true"

#data loader
loader = WebBaseLoader("https://docs.langchain.com/langsmith/data-storage-and-privacy")
input_data = loader.load()
print(f"Loaded {len(input_data)} documents.")

#data transformation
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(input_data)
print(f"Split into {len(texts)} chunks.")

# embeddings
embeddings = GoogleGenerativeAIEmbeddings(model ="gemini-embedding-001")

# vector store
vectorstoredb = FAISS.from_documents(texts, embeddings)
print("Vector store created.")

# retreival chain
retriever = vectorstoredb.as_retriever()
prompt = ChatPromptTemplate.from_messages({
   """
    Answer the question based on the context below.
        <Context>
        ${context}
        </Context>
    """
})
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
retriever_chain = create_retrieval_chain(retriever, document_chain)
result = retriever_chain.invoke({"input": "The Agent Server provides a durable execution runtime"})
print(result)




