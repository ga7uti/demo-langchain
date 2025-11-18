from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

# data ingestion
loader = PyPDFLoader("docs/sample.pdf")
documents = loader.load()

# data transformation
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50)
final_doc = text_splitter.split_documents(documents)
print(final_doc)

# data ingestion
# loader = TextLoader("docs/speech.txt")
# documents = loader.load()

# data transformation
# text_splitter = CharacterTextSplitter(
#     separator="\n",
#     chunk_size=100, 
#     chunk_overlap=20)
# final_doc = text_splitter.split_documents(documents)
# print(final_doc)




