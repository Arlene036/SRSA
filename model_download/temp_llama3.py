# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", 
                                             device_map="auto", 
                                             torch_dtype=torch.bfloat16
                                             )

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=256)
generated_outputs = outputs[0, input_ids['input_ids'].shape[-1]:]

print(tokenizer.decode(generated_outputs, skip_special_tokens=True))
