import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# data ingestion
loader = TextLoader("docs/speech.txt")
documents = loader.load()

# data transformation
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30)
final_doc = text_splitter.split_documents(documents)

# embeddings
embeddings = GoogleGenerativeAIEmbeddings(model ="gemini-embedding-001")

# vector store
vectorstore = InMemoryVectorStore.from_documents(final_doc, embeddings)
query = "Make sure you installed into the same Python interpreter"
docs = vectorstore.similarity_search(query)
print(docs)
