import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Ollama GenAI App"
os.environ["LANGSMITH_TRACING_V2"] = "true"

#prompt template
prompt = ChatPromptTemplate.from_messages({
   """
    You are an expert assistant. Provide detailed and accurate answers.
    Question: ${input}
    """
})

#streamlit
st.title("Ollama GenAI App")
input_text = st.text_input("What is your question?")

#initialize Ollama LLM
llm = Ollama(model="gemma3:1b")
outputParser = StrOutputParser()
chain = prompt | llm | outputParser

if input_text:
    response = chain.invoke({"input": input_text})
    st.write("### Response:")
    st.write(response)