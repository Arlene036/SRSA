from search_agent.search_agent import SearchAgent, SearchAgentOutput
from search_agent.rewrite_search import RewriteAgent, RewriteAgentHuggingface, SimpleSearchAgentOutput
from search_agent.offline_model import OfflineModel, OfflineModelHuggingface
from search_agent.simple_search_agent import SimpleSearchAgent, SimpleSearchAgentHuggingface
import os
import asyncio

os.environ["TAVILY_API_KEY"] = "YOUR_API_KEY"

MODEL = 'llama'
TYPE = 'simple_search'


async def get_rewrite_result():
    rewrite_agent = RewriteAgentHuggingface(model_name = MODEL)
    result = await rewrite_agent._onetime_run('what is string')
    print(result)

def get_simple_search_result():
    simple_search_agent = SimpleSearchAgentHuggingface(model_name = MODEL)
    result = simple_search_agent._react_run('what is string')
    print(result)

if __name__ == '__main__':
    if TYPE == 'rewrite':
        asyncio.run(get_rewrite_result())
    if TYPE == 'simple_search':
        get_simple_search_result()