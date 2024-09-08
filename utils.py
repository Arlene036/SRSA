import os
from typing import Dict
from langchain import agents
from langchain.base_language import BaseLanguageModel
from langchain_community.utilities import GoogleSearchAPIWrapper

# from tools.database_retrieval import *
from tools.online_search import *
from langchain.tools import Tool
from typing import Any, AsyncIterator, List, Literal

from fastapi import FastAPI
from langchain.agents import AgentExecutor, tool
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.tools.render import render_text_description
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.prompts import MessagesPlaceholder
from langchain_community.tools.convert_to_openai import format_tool_to_openai_tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_xml_agent
from langserve import add_routes
from langserve.pydantic_v1 import BaseModel
from tools.other_tools import *
from langchain_community.utilities import SerpAPIWrapper
from langchain_experimental.utilities import PythonREPL
# from tools.database_retrieval import SupabaseRetrieval
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
# from langchain_community.tools.tavily_search import TavilySearchResults, TavilyAnswer
from tools.tavily_search import TavilySearchResults, TavilyAnswer
from tools.search_agent_tool import SearchAgentTool
from operator import itemgetter
from prompts.search_prompt import SERP_SEARCH_TOOL_PROMPT
from tools.other_tools import Time
from langchain.memory import ConversationBufferMemory
from typing import List, Tuple
from langchain_core.agents import AgentAction
from typing import Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents import AgentOutputParser

google_search = GoogleSearchAPIWrapper()
google_search_tool =  Tool(
            name="Google_Search",
            description="Search Google for recent results.",
            func=google_search.run
        )

serp_params = {
    "engine": "google",
    "gl": "cn",
    "hl": "zh-cn",
}
serp_search = SerpAPIWrapper(params=serp_params)
serp_search_tool =  Tool(
            name="serp_search",
            description=SERP_SEARCH_TOOL_PROMPT,
            func=serp_search.run
        )

tavily_search_tool = TavilySearchResults(max_results=3, context_str_limit=800)

tools_map = {
    'serp_search': serp_search_tool,
    'tavily_search': tavily_search_tool,
    'google_search': google_search_tool,
    # 'supabase_rag': SupabaseRetrieval(),
    'search_agent': SearchAgentTool(),
    'time': Time()
}

def make_tools(tool_list:List[str] = ['supabase_rag', 'search_agent','time']):
    all_tools = []
    for tool in tool_list:
        all_tools.append(tools_map[tool])

    return all_tools

def convert_tools(tools):
    return "\n".join([f"{tool.name}: {tool.description}" for tool in tools])


def format_xml(
    intermediate_steps: List[Tuple[AgentAction, str]],
) -> str:
    """Format the intermediate steps as XML.

    Args:
        intermediate_steps: The intermediate steps.

    Returns:
        The intermediate steps as XML.
    """
    log = ""
    for action, observation in intermediate_steps:
        log += (
            f"<tool>{action.tool}</tool><function_call>{action.tool_input}"
            f"</function_call><observation>{observation}</observation>"
        )
    return log

class XMLAgentOutputParser(AgentOutputParser):
    """Parses tool invocations and final answers in XML format.

    Expects output to be in one of two formats.

    If the output signals that an action should be taken,
    should be in the below format. This will result in an AgentAction
    being returned.

    ```
    <tool>search</tool>
    <function_call>what is 2 + 2</function_call>
    ```

    If the output signals that a final answer should be given,
    should be in the below format. This will result in an AgentFinish
    being returned.

    ```
    <final_answer>Foo</final_answer>
    ```
    """

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        print('>>>>>>>>>original text>>>>>>>>>>>>>>>',text)
        if "</tool>" in text:
            tool, tool_input = text.split("</tool>", 1)
            _tool = tool.split("<tool>")[1]
            _tool_input = tool_input.split("<function_call>")[1]
            print('>>>>>>>>>>>>>>>>> tool_input 1 >>>>>>>>>>>>>>>>>>>>>',_tool_input)

            if "</function_call" in _tool_input:
                _tool_input = _tool_input.split("</function_call")[0]

                # multiple parameters
                if "&" in _tool_input:
                    _tool_input = _tool_input.split("&")
                    _tool_input: dict = {param.split("=")[0]: param.split("=")[1] for param in _tool_input}
                print('>>>>>>>>>>>>>>>>> tool_input 2 >>>>>>>>>>>>>>>>>>>>>',_tool_input)

            return AgentAction(tool=_tool, tool_input=_tool_input, log=text)
        
        elif "<final_answer>" in text:
            _, answer = text.split("<final_answer>")
            if "</final_answer>" in answer:
                answer = answer.split("</final_answer>")[0]
            return AgentFinish(return_values={"output": answer}, log=text)
        else:
            return AgentFinish(return_values={"output": text}, log=text)
            # raise ValueError("parser can not find the tool or final answer")

    def get_format_instructions(self) -> str:
        raise NotImplementedError

    @property
    def _type(self) -> str:
        return "xml-agent"

def get_memory_for_conversation(conversation_memories: Dict[str, ConversationBufferMemory], conversation_id: str) -> ConversationBufferMemory:
    if conversation_id not in conversation_memories:
        # Create a new memory instance for the conversation if it doesn't exist
        conversation_memories[conversation_id] = ConversationBufferMemory(return_messages=True, input_key='input', output_key='output')
    return conversation_memories[conversation_id]

def create_agent_with_tools_conv(llm, prompt, conversation_memories, conversation_id, tool_list=None) -> AgentExecutor:
    """Create an agent that uses XML to format its logic.
    With the memory for each conversation section
    """

    missing_vars = {"tools", "agent_scratchpad"}.difference(prompt.input_variables)
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")

    tools: List[BaseTool] = make_tools(tool_list=tool_list)
    if len(tools)>=1:
        llm_with_tools = llm.bind_tools(tools, stop=["</function_call>"])
    else:
        llm_with_tools = llm

    memory = get_memory_for_conversation(conversation_memories, conversation_id)
    prompt = prompt.partial(
        tools=render_text_description(tools),
    )
    agent = (
        {  
            "tools": lambda x: x['tools'],
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_xml(
                x["intermediate_steps"]
            ),
            "agent_name": lambda x: x["agent_name"],
            "user_name": lambda x: x["user_name"],
            "conversation_id": lambda x: x["conversation_id"]
        }
        | 
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
        )
        | prompt
        | llm_with_tools
        | XMLAgentOutputParser()
    )

    agent_executor = AgentExecutor(memory=memory, agent=agent, tools=tools, verbose=True).with_config(
        {"run_name": "agent"}
    )

    return agent_executor


def create_agent_with_tools_conv_openai(llm, prompt, conversation_memories, tool_list=None) -> AgentExecutor:
    """
    modify the memory for each conversation section
    """
    tools = make_tools(tool_list=tool_list)
    if len(tools)>=1:
        llm_with_tools = llm.bind_tools(tools)
    else:
        llm_with_tools = llm

    # memory = get_memory_for_conversation(conversation_memories, conversation_id)

    agent = (
        {  
            "tools": lambda x: x['tools'],
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
            "agent_name": lambda x: x["agent_name"],
            "user_name": lambda x: x["user_name"]
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )

    agent_executor = AgentExecutor( agent=agent, 
                                   tools=tools,
                                   max_iterations=3, 
                                   verbose=True).with_config(
        {"run_name": "agent"}
    )
    return agent_executor