#!/bin/bash

MODELS=(
    "deepseek-ai/DeepSeek-V3-0324"
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    "google/gemma-3-27b-it"
    "meta-llama/Llama-3.3-70B-Instruct"
    # "Qwen/Qwen3-235B-A22B"
)

PHRASES=(
    "This is a sample answer"
    "There is no correct answer"
)

MAX_TOKENS=4096
INFERENCE_PROVIDER="nebius"
NUM_EXAMPLES=0
OUTPUT_DIR="logs/responses"

for model in "${MODELS[@]}"; do
    for phrase in "${PHRASES[@]}"; do
        echo "Running evaluation for model: $model with phrase: '$phrase'"
        
        uv run gpqa-bs-cli-hf.py \
            --phrase "$phrase" \
            --model-name "$model" \
            --max-tokens "$MAX_TOKENS" \
            --inference-provider "$INFERENCE_PROVIDER" \
            --num-examples "$NUM_EXAMPLES" \
            --output-dir "$OUTPUT_DIR"
        
        echo "----------------------------------------"
    done
done

# INFERENCE_PROVIDER="novita"

# MODELS=(
#     "deepseek/deepseek-r1-0528-qwen3-8b"
#     "deepseek/deepseek-r1-0528"
# )
# MAX_TOKENS=3200

# for model in "${MODELS[@]}"; do
#     for phrase in "${PHRASES[@]}"; do
#         echo "Running evaluation for model: $model with phrase: '$phrase'"
        
#         uv run gpqa-bs-cli-hf.py \
#             --phrase "$phrase" \
#             --model-name "$model" \
#             --max-tokens "$MAX_TOKENS" \
#             --inference-provider "$INFERENCE_PROVIDER" \
#             --num-examples "$NUM_EXAMPLES" \
#             --output-dir "$OUTPUT_DIR"
        
#         echo "----------------------------------------"
#     done
# done


echo "All evaluations completed. Results saved to $OUTPUT_DIR"