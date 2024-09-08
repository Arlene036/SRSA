"""Tool for the Tavily search API."""

from typing import Dict, List, Optional, Type, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from transformers import Tool
# from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from tools.tool_utils import TavilySearchAPIWrapper

class TavilyInput(BaseModel):
    """Input for the Tavily tool."""

    query: str = Field(description="search query to look up")

def truncate_context(data: List[Dict], max_length=5000):
    """
    Truncates the text content in each dictionary of the given list
    if it exceeds the specified maximum length.

    Args:
    - data (List[Dict]): List of dictionaries where each dictionary contains 'url' and 'context' keys.
    - max_length (int): Maximum allowed length for the context string.

    Returns:
    - List[Dict]: Modified list with truncated context strings.
    """
    # data_copy = data.copy()
    # for i in range(len(data_copy)):
    #     if 'content' in data_copy[i]:
    #         context = data_copy[i]['content']
    #         if len(context) > max_length:
    #             data_copy[i]['content'] = context[:max_length]
    # return data_copy
    for item in data:
        if 'content' in item.keys():
            context = item['content']
            if len(context) > max_length:
                item['content'] = context[:max_length]
    return data

class TavilySearchResults(BaseTool):
    """Tool that queries the Tavily Search API and gets back json."""

    name: str = "tavily_search_results_limited"
    description: str = (
        "A search engine optimized for comprehensive, accurate, and trusted results. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query."
    )
    api_wrapper: TavilySearchAPIWrapper = Field(default_factory=TavilySearchAPIWrapper)
    max_results: int = 5
    search_depth: str = 'basic'
    context_str_limit: int = 5000
    args_schema: Type[BaseModel] = TavilyInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Union[List[Dict], str]:
        """Use the tool."""
        try:
            query = query.ljust(5, '?')
            results = self.api_wrapper.results(
                query,
                self.max_results,
                search_depth=self.search_depth,
            )
            truncated_results = truncate_context(results, self.context_str_limit)
            return truncated_results

        except Exception as e:
            return repr(e)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Union[List[Dict], str]:
        """Use the tool asynchronously."""
        try:
            query = query.ljust(5, '?')
            results = await self.api_wrapper.results_async(
                query,
                self.max_results,
                search_depth=self.search_depth
            )
            truncated_results = truncate_context(results, self.context_str_limit)
            return truncated_results
        except Exception as e:
            return repr(e)


class TavilyAnswer(BaseTool):
    """Tool that queries the Tavily Search API and gets back an answer."""

    name: str = "tavily_answer"
    description: str = (
        "A search engine optimized for comprehensive, accurate, and trusted results. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query. "
        "This returns only the answer - not the original source data."
    )
    max_results: int = 5
    api_wrapper: TavilySearchAPIWrapper = Field(default_factory=TavilySearchAPIWrapper)
    args_schema: Type[BaseModel] = TavilyInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Union[List[Dict], str]:
        """Use the tool."""
        try:
            query = query.ljust(5, '?')
            return self.api_wrapper.raw_results(
                query,
                max_results=self.max_results,
                include_answer=True,
                search_depth="advanced",
            )["answer"]
        except Exception as e:
            return repr(e)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Union[List[Dict], str]:
        """Use the tool asynchronously."""
        try:
            query = query.ljust(5, '?')
            result = await self.api_wrapper.raw_results_async(
                query,
                max_results=self.max_results,
                include_answer=True,
                search_depth="advanced",
            )
            return result["answer"]
        except Exception as e:
            return repr(e)


class TavilySearchHuggingfaceTool(Tool):
    """Tool that queries the Tavily Search API and gets back json."""

    name: str = "tavily_search_results"
    description: str = (
        "A search engine optimized for comprehensive, accurate, and trusted results. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query."
    )
    inputs = {
        "task": {
            "type": "text",
            "description": "the text input of search engine",
        }
    }
    output_type = "text"
    max_results: int = 5
    search_depth: str = 'basic'
    context_str_limit: int = 5000
    args_schema: Type[BaseModel] = TavilyInput

    def forward(
        self,
        query: str
    ) -> Union[List[Dict], str]:
        """Use the tool."""
        try:
            self.api_wrapper = TavilySearchAPIWrapper()
            query = query.ljust(5, '?')
            results = self.api_wrapper.results(
                query,
                self.max_results,
                search_depth=self.search_depth,
            )
            truncated_results = truncate_context(results, self.context_str_limit)
            return truncated_results

        except Exception as e:
            return repr(e)