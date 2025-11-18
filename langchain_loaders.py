from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader

loader = TextLoader("docs/speech.txt")
documents = loader.load()
print(documents)

# loader = PyPDFLoader("docs/sample.pdf")
# documents = loader.load()
# print(documents)

# loader = WebBaseLoader("https://en.wikipedia.org/wiki/Artificial_intelligence")
# documents = loader.load()
# print(documents)



