from langchain_core.runnables import RunnableLambda
from typing import Dict, List
# from langchain.memory import ConversationBufferMemory
from search_agent.search_agent import SearchAgent, SearchAgentOutput
from search_agent.rewrite_search import RewriteAgent, SimpleSearchAgentOutput

from search_agent.offline_model import OfflineModel
from search_agent.simple_search_agent import SimpleSearchAgent

from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from langsmith import Client
from langserve import add_routes
import datetime
import os

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = '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' 
os.environ["TAVILY_API_KEY"] = "YOUR_API_KEY"

client = Client()

RAW_CONTENT = False
MODEL = 'gemma'
CURRENT_TIME = datetime.datetime.now().strftime('%Y-%m-%d')

os.environ['LANGCHAIN_PROJECT'] = f"search_agent_{MODEL}_raw-content_{RAW_CONTENT}_{CURRENT_TIME}"

app = FastAPI()
search_agent = SearchAgent(model_name = MODEL, raw_content = RAW_CONTENT)
rewrite_agent = RewriteAgent(model_name = MODEL)
offlineModel = OfflineModel(model_name = MODEL)
simpleSearchAgent = SimpleSearchAgent(model_name = MODEL)

class Input(BaseModel):
    query: str
    conversation_id: int

async def run_search(input: Input) -> str:
    try:
        query = input['query']
        result: SearchAgentOutput = await search_agent._run(user_query=query) 

        return result.Result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def run_rewrite_search_test_mode(input: Input) -> str:
    try:
        result: SimpleSearchAgentOutput = await rewrite_agent._run(user_query=input['query'])
        return result.Result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def run_simple_search_test_mode(input: Input) -> str:
    try:
        result = await simpleSearchAgent._run(user_query=input['query'])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
async def run_offline_test_mode(input: Input) -> str:
    try:
        result = await offlineModel._run(user_query=input['query'])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



add_routes(
    app,
    RunnableLambda(run_search),
    path="/search"
)

add_routes(
    app,
    RunnableLambda(run_rewrite_search_test_mode),
    path="/rewrite_search_test_mode"
)

add_routes(
    app,
    RunnableLambda(run_simple_search_test_mode),
    path="/simple_search_test_mode"
)

add_routes(
    app,
    RunnableLambda(run_offline_test_mode),
    path="/offline_test_mode"
)

if __name__ == "__main__":
    uvicorn.run("server_search_agent:app", host="0.0.0.0", port=6006, reload=True)
