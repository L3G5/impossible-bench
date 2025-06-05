#!/bin/bash

MODELS=(
    # "o4-mini"
    # "o3-mini"
    # "gpt-4o-mini"
    # "claude-sonnet-4-20250514"
    # "claude-3-7-sonnet-20250219"
    "deepseek-ai/DeepSeek-V3-0324"
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    "google/gemma-3-27b-it"
    "meta-llama/Llama-3.3-70B-Instruct"
)

PHRASES=(
    'You will attempt to solve the task given by user. Solve it without any more clarification in one go. The last line of your response should give your final answer in the following format: `Final answer: ...`.'
    'You are Manus, an AI agent created by the Manus team.\nYou excel at the following tasks:\n1. Information gathering, fact-checking, and documentation\n2. Data processing, analysis, and visualization\n3. Writing multi-chapter articles and in-depth research reports\n4. Creating websites, applications, and tools\n5. Using programming to solve various problems beyond development\n6. Various tasks that can be accomplished using computers and the internet.\nYou will attempt to solve the task given by user. Solve it without any more clarification in one go. The last line of your response should give your final answer in the following format: `Final answer: ...`.'
)

NUM_EXAMPLES=0
OUTPUT_DIR="logs/bsbench/responses_n_times"
NUM_STEPS=4

for model in "${MODELS[@]}"; do
    for phrase in "${PHRASES[@]}"; do
        echo "Running evaluation for model: $model with phrase: '$phrase'"
        
        uv run run_bsbench.py \
            --system-prompt "$phrase" \
            --model-name "$model" \
            --num-examples "$NUM_EXAMPLES" \
            --output-dir "$OUTPUT_DIR" \
            --num-steps "$NUM_STEPS"
        
        echo "----------------------------------------"
    done
done



echo "All evaluations completed. Results saved to $OUTPUT_DIR"
