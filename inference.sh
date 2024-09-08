#!/bin/bash

model_names=("gemma" "mistral"  "llama" )
model_types=("rewrite_react_search" "search_agent" "simple_search")
log_dir="logs"
mkdir -p "$log_dir"

for model_name in "${model_names[@]}"; do
    for model_type in "${model_types[@]}"; do
        log_file="$log_dir/${model_name}_${model_type}_$(date +'%Y%m%d_%H%M%S').log"
        
        echo ">>>>>>>>>>>>>>>>>>>Running inference with model_name=$model_name and model_type=$model_type>>>>>>>>>>>>>>>>>>>>>>"
        echo "Log file: $log_file"
        
        python -m evaluation.inference_hug \
            --model_name "$model_name" \
            --model_type "$model_type" \
            --test 1 2>&1 \
            --result_dic "evaluation/results" | tee "$log_file"
        
        echo "Completed inference with model_name=$model_name and model_type=$model_type"
        echo "Log saved to: $log_file"
        echo ">>>>>>>>>>>>>>>>>>Done inference with model_name=$model_name and model_type=$model_type>>>>>>>>>>>>>>>>>>>>>>>"
    done
done