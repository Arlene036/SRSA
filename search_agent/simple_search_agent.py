# input: model_name (string), query
# output: response
# ReAct agent with Tavily search tool
from tools.tool_utils import TavilySearchAPIWrapper
from pydantic import BaseModel
from typing import Any, AsyncIterator, List, Literal, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.base import BasePromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from tools.other_tools import Time
from .models import *
from transformers.agents import ReactAgent, ReactCodeAgent
from transformers import Tool
from huggingface_hub import list_models
from tools.tavily_search import TavilySearchHuggingfaceTool
# Define a custom parser by extending the BaseParser class
from .rewrite_search import SimpleSearchAgentOutput
from prompts.search_prompt import *

class SimpleSearchAgent():
    def __init__(self, model_name = 'llama', raw_content = False):     
        # model
        self.llm = get_llm(model_name)

    async def simple_search(self, user_query) -> AsyncIterator[str]:
        
        # Create the agent
        search_tool = TavilySearchResults(max_results=2)
        tools = [search_tool, Time()]
        agent_executor = create_react_agent(self.llm, tools).with_config(
            {"run_name": "agent"}
        )
        
        async for event in agent_executor.astream_events(
            {
                "input": user_query,
            },
            version="v1",
        ):
            with open('event.log', 'a', encoding='utf-8') as log_file:
                log_file.write(str(event) + '\n')

            kind = event["event"]
            if kind == "on_chain_end":
                if (
                    event["name"] == "agent"
                ):  # matches `.with_config({"run_name": "Agent"})` in agent_executor
                    result = event['data'].get('output')['output']
                    yield result
    
    async def _run(self, user_query: str) -> str:
        result = await self.simple_search(user_query)
        return result


class SimpleSearchAgentHuggingface():
    def __init__(self, model_name = 'gemma', max_new_tokens=64):
        self.llm, self.tokenizer = get_llm_huggingface(model_name)
        self.llm_engine = get_huggingface_client(model_name)
        self.tavily_search = TavilySearchAPIWrapper(context_str_limit=800)
        self.generating_result_prompt = GENERATING_RESULT_PROMPT
        self.max_new_tokens = max_new_tokens
        if model_name == 'mistral':
            self.is_mistral = True
        else:
            self.is_mistral = False
    
    def get_response(self, input: str) -> str:
        input_ids = self.tokenizer(input, return_tensors="pt").to("cuda")

        outputs = self.llm.generate(**input_ids, max_new_tokens=self.max_new_tokens)
        generated_outputs = outputs[0, input_ids['input_ids'].shape[-1]:]

        result = self.tokenizer.decode(generated_outputs, skip_special_tokens=True)

        return result
    
    def _react_run(self, user_query: str) -> str:
        agent = ReactCodeAgent(tools = [TavilySearchHuggingfaceTool()], llm_engine = self.llm_engine,
                                system_prompt=SEARCHING_REACT_SYSTEM_PROMPT)
        result, step_observation_logs = agent.run(user_query, is_mistral = self.is_mistral)
        return result, step_observation_logs

    
    async def _onetime_run(self, user_query: str) -> str:
        direct_search_result = await self.tavily_search.results_async(user_query, 
                                                                        max_results=5)
                                                                    

        direct_search_reference = 'User Query: ' + user_query + '\n'
        direct_search_reference += f"Question: {user_query}\n"

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

        print('-'*20)
        print('SEARCH_RESULTS')
        print(direct_search_reference)
        print('-'*20)

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

    