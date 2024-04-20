from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from dotenv import load_dotenv
from langchain_fireworks import ChatFireworks
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
from langchain.agents import create_structured_chat_agent
import langchain_google_vertexai as verai
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate
)
from langchain.tools.render import render_text_description
def callmensa(input_text):
    api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=300)
    wiki=WikipediaQueryRun(api_wrapper=api_wrapper)
    search = TavilySearchAPIWrapper(tavily_api_key=os.getenv("TAVILY_API_KEY"))
    tavily= TavilySearchResults(api_wrapper=search)
    pubmed=PubmedQueryRun()
    tools=[pubmed,wiki,tavily]
    
    load_dotenv()
    
    apikey=os.getenv('FIREWORKS_API_KEY')
   
    llm = ChatFireworks(model="accounts/fireworks/models/mixtral-8x7b-instruct", api_key=apikey, max_tokens=300)
    from langchain.tools.render import render_text_description
    rendered_tools = render_text_description(tools)
   # llm = ChatGoogleGenerativeAI(model="gemini-pro",
   #                          temperature=0.1, safety_settings={
    #    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    #   HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
   #}, google_api_key=GOOGLE_API_KEY)

    prompt_template = f""" Your name is Mensa. You are a medical practitioner and specialize on questions
            regarding female menstruatual health , periods , symptoms related to it , its solutions , 
            diseases related to it and myths related to it.Answer the question as detailed as possible 
            from the given sources, make sure to provide all the details,don't provide the wrong answer to 
            things you do not know and you should not entertain any questions that are not related to female menstruation 
            , periods , symptoms related to it , its solutions , diseases related to it and myths related to it.\n\n 
            Make sure to use only the wiki pubmed tool for information and no other sources strictly.
              Summarize your answers within 150 words. Do not provide articles link but you can tell the sources wherever needed.
              Here are the names and descriptions for each tool:

        {rendered_tools}
    """
    # prompt = [ SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template=prompt_template)),
    #            MessagesPlaceholder(variable_name='chat_history', optional=True),
    #            HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')),
    #            MessagesPlaceholder(variable_name='agent_scratchpad')]
   
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            prompt_template),
        ("user", "{input}")
        
    ]
)
   


    chain = prompt | llm 
    answer=chain.invoke({"input": {input_text}})
    print(answer)
    return(answer.content)
    
    

def main():
    st.set_page_config("Mensa")
    st.header("Chat with Mensa powered by Langchain & Mistral AI 👩‍⚕️")

    user_question = st.text_input("Ask Questions about your Menstrual Health #NoShame", placeholder="Enter your queries here 🤗")
    st.sidebar.title("Hey your personalised Chatbot is ready")
    st.sidebar.divider()
    st.sidebar.subheader("Ask me anything about your Menstrual Health")
    st.sidebar.divider()
    if user_question:
        st.write(callmensa(user_question))


if __name__ == "__main__":
    main()
