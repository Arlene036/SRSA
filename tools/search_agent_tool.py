import os

from typing import Optional
from typing import Dict, List
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

from search_agent import SearchAgent, SearchAgentOutput


conversation_memories_dict: Dict[int, List[str]] = {}

RAW_CONTENT = False
MODEL = 'openai'

def get_memory_for_conversation(conversation_memories: Dict[int, List[str]], conversation_id: int) -> List[str]:
    if conversation_id not in conversation_memories:
        conversation_memories[conversation_id] = []
    return conversation_memories[conversation_id]

class SearchAgentTool(BaseTool):

    name: str = "search_agent_tool"
    description: str = (
        "Use for searching the internet and interact with the user. There are two parameters, query is the user's input, conversation_id is the id of the conversation."
    )

    async def _run(
        self,
        query: str,
        conversation_id: int = 0
    ) -> SearchAgentOutput:

        search_agent = SearchAgent(model = MODEL, raw_content = RAW_CONTENT)
        has_memory = False
        
        conversation_memory: List[str] = get_memory_for_conversation(conversation_memories_dict, conversation_id)
        if len(conversation_memory) > 0:
            query = conversation_memory + ['User: {}'.format(input['query'])]
            query = '\n'.join(query)
            has_memory = True

        result: SearchAgentOutput = await search_agent._run(user_query=query, has_memory = has_memory)

        if result.Action == "Further":
            conversation_memories_dict[conversation_id].append('User: {}'.format(input['query']))
            conversation_memories_dict[conversation_id].append('Assistant: {}'.format(result.Result))
            return result
        else:
            return result
        