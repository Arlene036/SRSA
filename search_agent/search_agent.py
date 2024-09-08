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
from transformers.agents import ReactAgent, ReactCodeAgent, ReactCodeSearchAgent, ReactJsonAgent
from transformers import Tool
from huggingface_hub import list_models
from tools.tavily_search import TavilySearchHuggingfaceTool
import re
import json
from search_agent.parser import AskUserParser, StrategySuggestionParser, RephraseParser, GeneratedQuestionsSeparatedListOutputParser
from prompts.search_prompt import *
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

class SearchAgentOutput(BaseModel):
    Strategy: Optional[str] # choose from ['Parallel', 'Planning', 'Direct', None]
    Action: str # choose from ['Further', 'Done']
    Result: Optional[str] # str or None
    Url: Optional[List[str]] # str or None
    Rerference: Optional[List[str]] # str or None
        
class SearchAgentHuggingface():
    rephrase_prompt = SEARCH_DIRECT_REPHRASE_PROMPT1
    generating_result_prompt = GENERATING_RESULT_PROMPT
    search_strategy_prompt = SEARCH_STRATEGY_CLASSIFY_PROMPT
    prompt_map = {
            'Parallel':SEARCH_PARALLEL_PROMPT,
            'Planning_suggestions': SEARCH_PLANNING_REACT_PROMPT_SUGGESTIONS_HUGGINGFACE,
            "Planning": SEARCH_PLANNING_REACT_PROMPT_HUGGINGFACE
        }
    rephrase_parser = RephraseParser()
    strategy_parser = StrategySuggestionParser()
    parallel_question_generate_parser = GeneratedQuestionsSeparatedListOutputParser()


    def __init__(self, model_name = 'gemma', max_new_tokens=1000):
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
    
    async def _run_parallel_search(self, query: str, suggestions: str) -> Tuple[str, List[str], List[str]]:
        url_list = []
        refer_content = []
        parallel_prompt = self.prompt_map['Parallel']
        raw_parallel_question_generate = self.get_response(parallel_prompt.format(input=query, suggestions = suggestions), max_new_tokens=128)
        print('>>>>>> raw parallel question >>>>>>')
        print(raw_parallel_question_generate)
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        generated_questions = self.parallel_question_generate_parser.parse(raw_parallel_question_generate)

        if len(generated_questions) == 0:
            return "Direct", [], []
        else:
            tasks = [self.tavily_search.results_async(key_word, 
                                                      max_results=5) for key_word in generated_questions]
            results = await asyncio.gather(*tasks, return_exceptions=True) # including raw content
    
            final_reference = ""
            for key_word, result in zip(generated_questions, results):
                final_reference += f"Question: {key_word}\nSearch Result:"
                for i, res in enumerate(result, 1):
                    for key, value in res.items():
                        if key == 'url':
                            url_list.append(value)
                        if key == 'content':
                            refer_content.append(value)
                        final_reference += f"{key}: {value}\n"
            return final_reference, url_list, refer_content

    
    async def _run_planning_search(self, query: str, suggestions: str):
        planing_react_prompt = ''
        if suggestions != '':
            planing_react_prompt = self.prompt_map['Planning_suggestions'].format(suggestions=suggestions, input=query)
        else:
            planing_react_prompt = self.prompt_map['Planning'].format(input=query)
        
        agent = ReactCodeSearchAgent(tools = [TavilySearchHuggingfaceTool()], 
                                     max_iterations = 5,
                                     llm_engine = self.llm_engine,
                                system_prompt=SEARCHING_REACT_SEARCH_AGENT_SYSTEM_PROMPT1)
        
        result, step_observation_logs, sm_logs = agent.run(planing_react_prompt, is_mistral=self.is_mistral)
        return result, step_observation_logs, sm_logs

    async def _onetime_run(self, user_query: str) -> str:
        refer_content = []
        ##################### I.search strategy classification ##################### 
        current_time, tomorrow_time = get_time()
        search_strategy_classify_prompt = self.search_strategy_prompt.format(input=user_query,current_time=current_time,tomorrow_time=tomorrow_time)
        raw_search_strategy = self.get_response(search_strategy_classify_prompt, max_new_tokens=128)
        print('>>>>>> raw search strategy >>>>>>')
        print(raw_search_strategy)
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        try:
            selected_strategy, suggestions = self.strategy_parser.parse(raw_search_strategy) # [strategy, suggestions]
        except Exception as e:
            self.logging.error(f"An error occurred doing strategy parsing: {e}")
            selected_strategy = 'Planning'
            suggestions = ''
            
        #########  II.search strategy #################
        parallel_reference = ''
        planning_reference = ''
        direct_search_reference = ''

        # Paralle
        if selected_strategy == "Parallel":
            parallel_reference, refer_url, refer_content = await self._run_parallel_search(user_query, suggestions)

        # Planning
        if selected_strategy == "Planning":
            planning_res, iteration_logs, sm_logs = await self._run_planning_search(user_query, suggestions)
            planning_reference = f"###Reference Answers###\n{planning_res}\n\n###First Search Result ###:\n{iteration_logs[0]}\n\n'##Compressed Search Results##'\n{sm_logs}\n\n"
            # return selected_strategy, planning_res, refer_content
            refer_content = planning_reference
            # return SearchAgentOutput(Strategy=selected_strategy, Action='Done', Result=planning_res, 
                                #  Url=refer_url, Rerference=refer_content)

        # Direct
        if selected_strategy == "Direct":

            rephrased_question = ''
            if suggestions != '':
                parts = suggestions.split("'")
                if len(parts) > 1:
                    rephrased_question = parts[1]
            
            if rephrased_question == '':
                rephrase_p = self.rephrase_prompt.format(input=user_query)
                rephrased_question = self.get_response(rephrase_p, max_new_tokens=64)

            direct_search_result = await self.tavily_search.results_async(rephrased_question, 
                                                                          max_results=5,
                                                                         )

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

        ######## III.for whole reference if planning or direct ######################
        FINAL_REFERENCE = f"""You should give the answer to this question: {user_query}.\n\n
        Some potential questions and answers for reference are as follow:\n{parallel_reference}{planning_reference}{direct_search_reference}
        Again, question is {user_query}.\n\n
        """

        generating_result_prompt = self.generating_result_prompt.format(question=user_query, context=FINAL_REFERENCE)
        final_result = self.get_response(generating_result_prompt, max_new_tokens=1000)
        return selected_strategy, final_result, refer_content