# input: model_name
# output: langchain llm 类

# reference: 
# https://api.python.langchain.com/en/latest/chat_models/langchain_huggingface.chat_models.huggingface.ChatHuggingFace.html

from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from huggingface_hub import login
from huggingface_hub import InferenceClient
from typing import Callable
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


NAMES = {
    'gemma' : 'google/gemma-2-2b-it',
    'llama' : 'meta-llama/Meta-Llama-3-8B-Instruct',
    'mistral' : 'mistralai/Mistral-7B-Instruct-v0.3'
}

def get_llm(model_name):
    llm = HuggingFaceEndpoint(
            repo_id=NAMES[model_name], 
            task="text-generation",
            max_new_tokens=512, 
            do_sample=False, 
            repetition_penalty=1.03
        )

    chat = ChatHuggingFace(llm=llm, verbose=True)
    return chat

def get_llm_huggingface(model_name):
    tokenizer = AutoTokenizer.from_pretrained(NAMES[model_name])
    model = AutoModelForCausalLM.from_pretrained(
                NAMES[model_name],
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
    return model, tokenizer

def get_huggingface_client(model_name) -> Callable:
    client = InferenceClient(model=NAMES[model_name])

    def llm_engine(messages, stop_sequences=["Task"]) -> str:
        response = client.chat_completion(messages, stop=stop_sequences, max_tokens=1000)
        answer = response.choices[0].message.content
        return answer
    
    return llm_engine


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Callable, List

def get_huggingface_client1(model_name: str) -> Callable:
    # Load model and tokenizer locally
    tokenizer = AutoTokenizer.from_pretrained(NAMES[model_name])
    model = AutoModelForCausalLM.from_pretrained(
                NAMES[model_name],
                device_map="auto",
                torch_dtype=torch.bfloat16
            )

    def llm_engine(messages: List[str], stop_sequences: List[str] = ["Task"]) -> str:
        system_prompts = "\n".join([message["content"] for message in messages if message["role"] == "system"])
        user_messages = "\n".join([message["content"] for message in messages if message["role"] != "system"])
        
        input_text = f"{system_prompts}\n{user_messages}"
        
        input_ids = tokenizer(input_text, return_tensors="pt").to('cuda')

        # Generate outputs
        outputs = model.generate(**input_ids, max_new_tokens=1000)
        generated_outputs = outputs[0, input_ids['input_ids'].shape[-1]:]

        # Decode and return the result
        result = tokenizer.decode(generated_outputs, skip_special_tokens=True)
        
        # Optionally, handle stop sequences if necessary
        # This is a simplified version and may not perfectly handle stop sequences
        for stop_seq in stop_sequences:
            if stop_seq in result:
                result = result.split(stop_seq)[0]
        
        return result

    return llm_engine

