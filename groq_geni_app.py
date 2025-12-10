import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

#llm initialization
llm=ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

#prompt template
generic_template = "Translate the following English text to {language}:"

messages = [
    SystemMessage(content="Translate the following English text to French."),
    HumanMessage(content="Hello, how are you?")
]

prompt  = ChatPromptTemplate.from_messages(
    [
        ("system", generic_template),
        ("user", "{text}")
    ]
)

#parser initialization
parser = StrOutputParser()

#lcel 
chain = prompt| llm | parser
response = chain.invoke({"language":"French","text":"Hello"})
print("Response:", response)

