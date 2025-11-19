import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import uuid7

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGSMITH_TRACING_V2"] = "true"

id = uuid7()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# result=llm.invoke("What is LangChain? Explain in a concise manner.")
# print(result.content)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert AI engineer.Provide me answers to my questions accurately and concisely."),
    ("user", "${question}")
])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser
result = chain.invoke({"question": "Can you tell me about langsmith?"})
print(result)


