from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from langserve import add_routes

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

#llm initialization
llm=ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

#prompt template
generic_template = "Translate the following English text to {language}:"
prompt  = ChatPromptTemplate.from_messages(
    [
        ("system", generic_template),
        ("user", "{text}")
    ]
)

#parser initialization
parser = StrOutputParser()

#chain
chain = prompt| llm | parser

#App definition
app = FastAPI(title="Simple Groq App",
              version="1.0.0",
              description="A simple FastAPI app using Groq LLM to translate text."
              )

#Add routes
add_routes(
    app,
    chain,
    path="/translate",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)