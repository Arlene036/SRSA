# Strategy-Router Search Agent

We construct a search agent to improve the human-chatbot conversation quality, especially focus on long contextual queries.

## Installation

This project uses Python 3.9, which you can install using [Anaconda](https://www.anaconda.com/products/distribution). 

To create a new Anaconda environment with Python 3.9, use the following command:

```{bash}
conda create -n search-agent python==3.9
pip install -r requirements.txt
```

API needed: Huggingface API(model download and inference), Tavily search API, OpenAI API(evaluation part)


```{bash}
cd model_download
python temp_gemma.py
python temp_llama3.py
temp_mistral.py
```

By running these three files, models could be downloaded and tested.

## Adjust Huggingface Package

find this file:
`/opt/anaconda3/envs/search-agent/lib/python3.10/site-packages/transformers/agents/agents.py`
and replace it with this `agent/agents.py`

It basically add a new ReAct Agent especially for online search.

## Inference

```{bash}
bash inference.sh
```

Adjust the augment 'test' to be 0, then all samples in dataset will be inferred by all models and agent types.

## Evaluation

For LLM automatic evaluation, use the following file:
`evaluation/checker/checker_api.ipynb`

The inference is quick, takes two hours. After get the automatic evaluation result, use `evaluation/checker/checker_analysis.ipynb` to plot results.

