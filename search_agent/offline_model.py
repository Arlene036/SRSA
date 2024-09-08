# input: model_name, query
# output: response

from langchain_core.prompts import ChatPromptTemplate
from .models import *
from prompts.default_prompts import *
from prompts.search_prompt import *
from typing import List

class OfflineModel():
    def __init__(self, model_name = 'llama'):
        # model
        self.llm = get_llm(model_name)
        # prompt
        self.generating_result_prompt = ChatPromptTemplate.from_template(SIMPLE_OFFLINE_SEARCH_PROMPT)


    async def _run(self, user_query: str) -> str:
        chain = self.generating_result_prompt | self.llm
        result = await chain.ainvoke({'question': user_query})
        return result
    
class OfflineModelHuggingface():
    def __init__(self, model_name = 'gemma'):
        self.llm, self.tokenizer = get_llm_huggingface(model_name)
    
    def _run(self, user_query: str) -> str:
        input_ids = self.tokenizer(user_query, return_tensors="pt").to("cuda")

        outputs = self.llm.generate(**input_ids, max_new_tokens=256)
        generated_outputs = outputs[0, input_ids['input_ids'].shape[-1]:]

        result = self.tokenizer.decode(generated_outputs, skip_special_tokens=True)
        return result

    def _run_batch(self, user_queries: List[str]) -> List[str]:
        encoding = self.tokenizer(user_queries, padding=True, return_tensors='pt').to('cuda')
        with torch.no_grad():
            outputs = self.llm.generate(**encoding,  max_new_tokens=256)
        results_list = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return results_list