# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 16:49:31 2024

@author: yash
"""
import streamlit as st
#from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from langserve import add_routes
from dotenv import load_dotenv
load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")
## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
# Initialize the LLM
llm = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key)

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableSequence

# Set up Streamlit interface
st.title("Langchain with Llama3 Model")
input_text = st.text_input("translate english to marathi?")

if input_text:
    # Create the messages for the prompt
    system_message = SystemMessage(content="Translate the following from English to Marathi")
    user_message = HumanMessage(content=input_text)
    
    # Invoke the model with the messages
    result = llm.invoke([system_message, user_message])
    
    # Parse the result using StrOutputParser
    parser = StrOutputParser()
    parsed_result = parser.invoke(result)
    
    # Display the parsed result in Streamlit
    st.write(parsed_result)