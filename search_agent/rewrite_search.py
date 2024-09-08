from pydantic import BaseModel
from typing import Dict, List, Tuple, Optional
from langchain.output_parsers import ListOutputParser
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.base import BasePromptTemplate
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from tools.tool_utils import TavilySearchAPIWrapper
from tools.tool_utils import create_react_agent_with_suggestions
from .models import *
import re
import json
from search_agent.parser import AskUserParser, StrategySuggestionParser, RephraseParser, GeneratedQuestionsSeparatedListOutputParser
from prompts.search_prompt import *
from transformers import pipeline
from transformers.agents import ReactAgent, ReactCodeAgent, ReactCodeSearchAgent
from transformers import Tool
from huggingface_hub import list_models
from tools.tavily_search import TavilySearchHuggingfaceTool
import asyncio
from fastapi import HTTPException
# from sentence_transformers import SentenceTransformer
import openai
import re
import logging
import traceback

# Define a custom parser by extending the BaseParser class

def get_time():
    import datetime
    current_time = datetime.datetime.now()
    tomorrow_time = current_time + datetime.timedelta(days=1)

    current_time_str = current_time.strftime('%Y %b %d')
    tomorrow_time_str = tomorrow_time.strftime('%Y %b %d')

    return current_time_str, tomorrow_time_str

class SimpleSearchAgentOutput(BaseModel):
    Result: Optional[str] # str or None
    Url: Optional[List[str]] # str or None
    Rerference: Optional[List[str]] # str or None

class RewriteAgent():
    """
    Input -> 
    understanding question and planning (chain of thoughts and few shot - for prompt choosing/generating) -> 
    [OPT]ask user or not -> 
    according to search strategy, generate questions tree (HyDE & Query Expansion), include domain recoginization: choose one of them{'general','news'} ->
    search and reranker -> 
    summarize each (each search result go for a llm call, finding most related snippet) -> 
    combine each ->
    [OPT]self-consistency check -> 
    for whole reference
    """
    tavily_search: any
    generating_result_prompt: BasePromptTemplate
    rephrase_prompt: BasePromptTemplate
    rephrase_parser: BaseOutputParser

    def __init__(self, model_name = 'llama', raw_content = False):
        
        # model
        self.llm = get_llm(model_name)

        # search parameter
        self.raw_content = raw_content
        self.tavily_search = TavilySearchAPIWrapper(context_str_limit=800)
        # self.sentence_transformer = SentenceTransformer('paraphrase-MiniLM-L6-v2') # TODO？

        # prompt
        self.generating_result_prompt = ChatPromptTemplate.from_template(GENERATING_RESULT_PROMPT)
        self.rephrase_prompt = ChatPromptTemplate.from_template(SEARCH_DIRECT_REPHRASE_PROMPT)

        # parser
        self.rephrase_parser = RephraseParser()


    async def _run(self, user_query: str) -> str:

        refer_url = []

        rephrased_question = ''

        rephrase_chain = (
                self.rephrase_prompt | 
                self.llm |
                self.rephrase_parser
            )
        rephrased_question: str = await rephrase_chain.ainvoke({'input': user_query})
        
        direct_search_result = await self.tavily_search.results_async(rephrased_question, 
                                                                        max_results=5,
                                                                        include_raw_content=self.raw_content)

        direct_search_reference = 'User Query: ' + user_query + '\n'
        direct_search_reference += f"Rephrased Question: {rephrased_question}\n"

        refer_url = []
        refer_content = []
        for i, result in enumerate(direct_search_result, 1):
            direct_search_reference += f"\nResult {i}:\n"
            for key, value in result.items():
                if key == 'url':
                    refer_url.append(value)
                if key == 'content':
                    refer_content.append(value)
                direct_search_reference += f"{key}: {value}\n"

        ######## III.for whole reference ######################
        FINAL_REFERENCE = f"""User's original question is {user_query}.
        Some potential questions and answers for reference are as follow:\n{direct_search_reference}
        """
        rag_chain = (
            self.generating_result_prompt | 
            self.llm |
            StrOutputParser()
        )

        final_result = await rag_chain.ainvoke({'question': user_query,'context': FINAL_REFERENCE})
        return SimpleSearchAgentOutput(Result=final_result, 
                                 Url=refer_url, Rerference=refer_content)


class RewriteAgentHuggingface():
    rephrase_prompt = SEARCH_DIRECT_REPHRASE_PROMPT1
    generating_result_prompt = GENERATING_RESULT_PROMPT
    rephrase_parser = RephraseParser()

    def __init__(self, model_name = 'gemma', max_new_tokens=64):
        self.model_id = NAMES[model_name]
        self.llm, self.tokenizer = get_llm_huggingface(model_name)

        self.tavily_search = TavilySearchAPIWrapper(context_str_limit=800)
        self.max_new_tokens = max_new_tokens

        if model_name == 'llama':
            self.llm_engine = get_huggingface_client(model_name)
        else:
            self.llm_engine = get_huggingface_client1(model_name)
            
        if model_name == 'mistral':
            self.is_mistral = True
        else:
            self.is_mistral = False

        self.logging = logging.getLogger(self.__class__.__name__)
        self.logging.setLevel(logging.DEBUG)
    
    def get_response(self, input: str, max_new_tokens = 1000) -> str:
        input_ids = self.tokenizer(input, return_tensors="pt").to("cuda")

        outputs = self.llm.generate(**input_ids, max_new_tokens=max_new_tokens)
        generated_outputs = outputs[0, input_ids['input_ids'].shape[-1]:]

        result = self.tokenizer.decode(generated_outputs, skip_special_tokens=True)

        return result

    def _react_run(self, user_query: str) -> str:
        raw_rephrased_question = self.get_response(self.rephrase_prompt + '\n' + user_query, max_new_tokens=64)
        try:
            rephrased_question = self.rephrase_parser.parse(raw_rephrased_question)
        except Exception as e:
            self.logging.error(f"An error occurred when rephrasing question: {e}")
            rephrased_question = user_query

        agent = ReactCodeSearchAgent(tools = [TavilySearchHuggingfaceTool()], 
                                     max_iterations = 3,
                                     llm_engine = self.llm_engine,
                                system_prompt=SEARCHING_REACT_SYSTEM_PROMPT)
        result, step_observation_logs = agent.run(rephrased_question, is_mistral=self.is_mistral)
        return result, step_observation_logs


    async def _onetime_run(self, user_query: str) -> str:
        raw_rephrased_question = self.get_response(self.rephrase_prompt + '\n' + user_query, max_new_tokens=64)
        try:
            rephrased_question = self.rephrase_parser.parse(raw_rephrased_question)
        except Exception as e:
            self.logging.error(f"An error occurred when rephrasing question: {e}")
            rephrased_question = user_query

        ### search results for each sub-questions ###
        direct_search_result = await self.tavily_search.results_async(rephrased_question, 
                                                                        max_results=5)
                                                                    

        direct_search_reference = 'User Query: ' + user_query + '\n'
        direct_search_reference += f"Rephrased Question: {rephrased_question}\n"

        refer_url = []
        refer_content = []
        for i, result in enumerate(direct_search_result, 1):
            direct_search_reference += f"\nResult {i}:\n"
            for key, value in result.items():
                if key == 'url':
                    refer_url.append(value)
                if key == 'content':
                    refer_content.append(value)
                direct_search_reference += f"{key}: {value}\n"

        ###for whole reference ####
        FINAL_REFERENCE = f"""User's original question is {user_query}.
        Some potential questions and answers for reference are as follow:\n{direct_search_reference}
        """

        final_user_prompt = self.generating_result_prompt.format(context = FINAL_REFERENCE, 
                                                                 question = user_query)
        final_answer = self.get_response(final_user_prompt)

        
        # return SimpleSearchAgentOutput(Result=final_answer, 
        #                          Url=refer_url, Rerference=refer_content)
        return final_answer, refer_content

        