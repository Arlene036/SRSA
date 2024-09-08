# NOTICE: This file is to alternated by the "search-agent" tool.
from typing import Optional
from langchain.tools import Tool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_community.utilities import SerpAPIWrapper
from typing import Any, AsyncIterator, List, Literal
from langchain.prompts import ChatPromptTemplate
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import ListOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts import PromptTemplate
from prompts.search_prompt import SEARCH_Q_GEN_PROMPT
import re
# from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from tools.tool_utils import TavilySearchAPIWrapper
from prompts.search_prompt import HIERARCHY_SEARCH_PROMPT

# serp
# refer to https://serpapi.com/search-api
params = {
    "engine": "bing",
    "gl": "cn",
    "hl": "zh-cn",
}
serp_search = SerpAPIWrapper(params=params)
tavily_search = TavilySearchAPIWrapper(context_str_limit=800)

# google
google_search = GoogleSearchAPIWrapper()
search_tool_google =  Tool(
            name="Google_Search",
            description="Search Google for recent results.",
            func=google_search.run
        )

class LineSeparatedListOutputParser(ListOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "output_parsers", "list"]

    def get_format_instructions(self) -> str:
        return (
            "输出思路和生成的问题，每个问题换行。"
            "format例子: 思路：用户的思路\n生成的问题：\n1.问题1\n2.问题2"
        )

    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call and extract questions.

        Args:
            text (str): The text output from an LLM call, containing sections and questions.

        Returns:
            List[str]: A list of strings, each representing a found question in the format 'number.question'.
        """
        parts = text.split("生成的问题：")
        if len(parts) > 1:
            questions = parts[1].strip().split("\n")
            questions = [q for q in questions if q]

            # Extract questions starting with numbered sequence
            seq_questions = []
            for q in questions:
                if q.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                    seq_questions.append(q)
                else:
                    break  # Stop extracting if a question doesn't start with numbered sequence
            
            return seq_questions[:3]  # Return the first 3 questions starting with numbered sequence

        else:
            return []

    @property
    def _type(self) -> str:
        return "comma-separated-list"

class SarchingModel():
    def __init__(self) -> None:

        llm_for_searching = ChatOpenAI()

        output_parser = LineSeparatedListOutputParser()
        format_instructions = output_parser.get_format_instructions()
        
        searching_prompt = PromptTemplate(
            template= SEARCH_Q_GEN_PROMPT,
            input_variables=["input"],
            partial_variables={"format_instructions": format_instructions},
        )

        self.key_words_searching_chain = searching_prompt | llm_for_searching | output_parser
    
    def run(self, query: str):
        return self.key_words_searching_chain.invoke({"input": query})


class HierarchicalSearch(BaseTool):

    name: str = "hierachical_search"
    description: str = (
       HIERARCHY_SEARCH_PROMPT
    )

    def _get_key_words(self, json_questions) -> List[str]:
        key_words = []
        for key in json_questions:
            if key.startswith('question'):
                key_words.extend(json_questions[key])
        return key_words
    
    async def _run_async(self, query: str) -> str:
        # json_questions = search_model.run(query) # 这里企图用jsonOutputParser输出一个json
        # key_words_str_list = self._get_key_words(json_questions)
        key_words_str_list = search_model.run(query)
        if len(key_words_str_list) == 0:
            return 'No reference'
        
        tasks = [serp_search.arun(key_word) for key_word in key_words_str_list]
        results = await asyncio.gather(*tasks)
        
        final_reference = ""
        for key_word, result in zip(key_words_str_list, results):
            final_reference += f"Question: {key_word}\nSearch Result: {result}\n"

        return final_reference

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
       reference = asyncio.run(self._run_async(query))
       direct_answer = serp_search.run(query)

       final_res = f"""User's original question is {query}. 
       Some potential questions and answers for reference are as follow:\n{reference}\nNow goes to the original question.
       User's Original Question: {query}
       Direct Search Result: {direct_answer}.
       """
       return final_res
    

class HierarchicalSearch_Tavily(BaseTool):

    name: str = "hierachical_search_tavily"
    description: str = (
        HIERARCHY_SEARCH_PROMPT
    )

    def _get_key_words(self, json_questions) -> List[str]:
        key_words = []
        for key in json_questions:
            if key.startswith('question'):
                key_words.extend(json_questions[key])
        return key_words
    
    async def _run_async(self, query: str) -> str:

        key_words_str_list = search_model.run(query)
        if len(key_words_str_list) == 0:
            return 'No reference'

        tasks = [tavily_search.results_async(key_word, max_results=2) for key_word in key_words_str_list]
        results = await asyncio.gather(*tasks)
        
        final_reference = ""
        for key_word, result in zip(key_words_str_list, results):
            final_reference += f"Question: {key_word}\nSearch Result:"

            for i, res in enumerate(result, 1):
                # final_reference += f"\nResult {i}:\n"
                for key, value in res.items():
                    final_reference += f"{key}: {value}\n"

        return final_reference

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
       reference = asyncio.run(self._run_async(query))
       list_dict_answer = tavily_search.results(query, max_results=2)

       final_res = f"""User's original question is {query}. 
       Some potential questions and answers for reference are as follow:\n{reference}\nNow goes to the original question.
       User's Original Question: {query}
       Direct Search Result:
       """

       for i, result in enumerate(list_dict_answer, 1):
            final_res += f"\nResult {i}:\n"
            for key, value in result.items():
                final_res += f"{key}: {value}\n"

       return final_res

# init model
search_model = SarchingModel(model='openai')
# search_model = SarchingModel()