#!/bin/bash

MODELS=(
    "o4-mini"
    "o3-mini"
    "gpt-4o-mini"
    "claude-sonnet-4-20250514"
    "claude-3-7-sonnet-20250219"
)

PHRASES=(
    "This is a sample answer"
    "There is no correct answer"
)

NUM_EXAMPLES=0
OUTPUT_DIR="logs/responses"

for model in "${MODELS[@]}"; do
    for phrase in "${PHRASES[@]}"; do
        echo "Running evaluation for model: $model with phrase: '$phrase'"
        
        uv run gpqa-bs-cli-anthropic-openai.py \
            --phrase "$phrase" \
            --model-name "$model" \
            --num-examples "$NUM_EXAMPLES" \
            --output-dir "$OUTPUT_DIR"
        
        echo "----------------------------------------"
    done
done



echo "All evaluations completed. Results saved to $OUTPUT_DIR"