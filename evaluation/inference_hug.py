import pandas as pd
import time
from search_agent.search_agent import SearchAgentHuggingface
from search_agent.rewrite_search import RewriteAgent, RewriteAgentHuggingface, SimpleSearchAgentOutput
from search_agent.offline_model import OfflineModel, OfflineModelHuggingface
from search_agent.simple_search_agent import SimpleSearchAgent, SimpleSearchAgentHuggingface
import os
import asyncio
from datetime import datetime
from typing import List
import csv
import csv
import argparse

os.environ["TAVILY_API_KEY"] = "API" 

def read_existing_questions(filepath):
    existing_questions = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                existing_questions.add((row[0], row[1], int(row[2])))
    except FileNotFoundError:
        print("Results file not found, creating a new one.")
    return existing_questions

def get_rewrite_result(model_name, query):
    rewrite_agent = RewriteAgentHuggingface(model_name = model_name)
    result, refer_content = rewrite_agent._react_run(query)
    return result, refer_content

def get_react_result(model_name, query):
    simple_search_agent = SimpleSearchAgentHuggingface(model_name)
    result, refer_content = simple_search_agent._react_run(query)
    return result, refer_content

def get_simple_search_onetime_result(model_name, query):
    simple_search_agent = SimpleSearchAgentHuggingface(model_name)
    result, refer_content = simple_search_agent._onetime_run(query)
    return result, refer_content

def get_offline_model_result(model_name, query):
    offline_model = OfflineModelHuggingface(model_name = model_name)
    result = offline_model._run(query)
    return result, ''
 
async def get_search_agent_result(model_name, query):
    search_agent = SearchAgentHuggingface(model_name = model_name)
    selected_strategy, result, refer_content = await search_agent._onetime_run(query)
    return selected_strategy, result, refer_content

map_function_baseline = {
    'offline_model': get_offline_model_result, # --
    'rewrite_react_search': get_rewrite_result, # ---
    'react_search': get_react_result, 
    'simple_search': get_simple_search_onetime_result, # ---
    'search_agent': get_search_agent_result # ---
}

#####################################################################
async def inference(model_name, model_type, questions_df, results_file, test=0):
    f = map_function_baseline[model_type]

    existing_triples = set()

    # Ensure the directory for results_file exists
    results_dir = os.path.dirname(results_file)
    os.makedirs(results_dir, exist_ok=True)
    
    if not os.path.exists(results_file):
        with open(results_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Model Type', 'Model Name', 'test_id', 'query', 'result', 'reference', 'strategy'])
    else:
        existing_triples = read_existing_questions(results_file)

    count = 0
    with open(results_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for idx, row in questions_df.iterrows():
            question = row['query']
            test_id = idx
            current_triple = (model_type, model_name, test_id)
            if current_triple not in existing_triples:
                selected_strategy = ''
                if model_type != 'search_agent':
                    result, refer_content = f(model_name, question)
                else:
                    selected_strategy, result, refer_content = await f(model_name, question)
                writer.writerow([model_type, model_name, test_id, question, result, refer_content, selected_strategy])
                count += 1
                if test == 1 and count > 2:
                    break
                print(f'Test count: {count}')
            else:
                print(f'current triple has been tested: {current_triple}')

async def inference1(model_name, model_type, questions_df, results_file, test=0):
  
    existing_triples = set()
    
    results_dir = os.path.dirname(results_file)
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(results_file):
        with open(results_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Model Type', 'Model Name',  'test_id', 'query', 'result', 'reference', 'strategy'])
    else:
        existing_triples = read_existing_questions(results_file)

    count = 0
    with open(results_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for idx, row in questions_df.iterrows():
            question = row['Query']
            test_id = idx
            current_triple = (model_type, model_name, test_id)
            
            print(f'==========current triple========: {current_triple}')
            print(f'==========existing triples========: {existing_triples}')
            print(f'==========current triple not in existing triples========: {current_triple not in existing_triples}')

            if current_triple not in existing_triples:
                if model_type == 'rewrite_react_search' or  model_type == 'simple_search':
                    rewrite_agent = RewriteAgentHuggingface(model_name=model_name)
                elif model_type == 'search_agent':
                    search_agent = SearchAgentHuggingface(model_name=model_name)

                selected_strategy = ''
                if model_type == 'simple_search':
                    result, refer_content = await rewrite_agent._onetime_run(question)
                elif model_type == 'rewrite_react_search':
                    result, refer_content = rewrite_agent._react_run(question)
                elif model_type == 'search_agent':
                    selected_strategy, result, refer_content = await search_agent._onetime_run(question)

                writer.writerow([model_type, model_name, test_id, question, result, refer_content, selected_strategy])
                count += 1
                if test == 1 and count > 5:
                    # go into checker
                    break
                print(f'Test count: {count}')

############### Main ##############
if __name__ == '__main__':
    # args: model_name, model_type, questions_df(.csv file path), results_dic(.csv file path)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama', choices = ['llama', 'gemma','mistral'])
    parser.add_argument('--model_type', type=str, choices=['rewrite_react_search', 'simple_search',
                                                           'search_agent'], default='simple_search')
    parser.add_argument('--questions_df', type=str, default='evaluation/test_dataset/CQED_new.csv')
    parser.add_argument('--results_dic', type=str, default='evaluation/results')
    parser.add_argument('--test', type=int, default=0)
    args = parser.parse_args()

    if args.test == 1: # test mode
        args.results_dic += '/sample_test'

    questions_df = pd.read_csv(args.questions_df)
    print('args:', args)
    print('>>>>>>> Start inference >>>>>>>')


    asyncio.run(inference1(
        args.model_name, 
        args.model_type, 
        questions_df,
        args.results_dic + '/test_results.csv', # args.results_dic + '/' + args.model_name + '/' + args.model_type + '/' + t + '/results.csv', 
        args.test
    ))

