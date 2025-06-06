# Impossible Bench

## How to run gpqa-bs

We use inference providers for HuggingFace and Batch API for OpenAI and Anthropic.

### HF inference providers

You will need to set your `HUGGINGFACE_API_KEY` in `.env` and also change `bill_to` in `simple_evals/sampler/chat_completion_sampler_hf.ChatCompletionSamplerHF`. Then ti run a custom evaluation use:

```bash
uv run gpqa-bs-cli-hf.py --phrase "There is no correct answer" \
                 --model-name "meta-llama/Llama-3.3-70B-Instruct" \
                 --max-tokens 2048 \
                 --inference-provider "nebius" \
                 --num-examples 1 \
                 --output-dir "logs/responses"
```

To repeat experiments in the paper use:

```bash
./run-hf-gpqa.sh
```

### OpenAI and Anthropic

```bash
uv run gpqa-bs-cli-anthropic-openai.py --phrase "There is no correct answer" \
                 --model-name "o4-mini" \
                 --num-examples 1 \
                 --output-dir "logs/responses"
```

To repeat experiments in the paper use:

```bash
./run-openai-anthropic-gpqa.sh
```

### Logs 

are availible at `logs/responses` 

## how to get BSBench data

Extract data from data.zip (for example, `7za x data.zip`) using password `LgvnmKvpgKbriiGvng`.

### How to run BSBench

To repeat experiments in the paper use `./run_bsbench.sh`

To run your own eperiments use 

```bash
        uv run run_bsbench.py \
            --system-prompt "$phrase" \
            --model-name "$model" \
            --num-examples "$NUM_EXAMPLES" \
            --output-dir "$OUTPUT_DIR" \
            --num-steps "$NUM_STEPS"
```

### Logs

are availible at `logs/bsbench/responses` with the initial dataset and `logs/bsbench/responses_n_times` with the final dataset
