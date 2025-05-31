import click
from simple_evals.gpqa_bs_eval import GPQAEvalBS
from simple_evals.sampler.chat_completion_sampler_hf import ChatCompletionSamplerHF
import json
import os
import dotenv
from datetime import datetime

dotenv.load_dotenv()

@click.command()
@click.option('--phrase', required=True, help='The phrase to test')
@click.option('--model-name', required=True, help='Model name to evaluate')
@click.option('--max-tokens', default=4096, help='Maximum tokens for generation')
@click.option('--inference-provider', default='nebius', help='Inference provider to use')
@click.option('--num-examples', type=click.INT, default=None, help='Number of examples to evaluate (None for all)')
@click.option('--output-dir', default='logs/responses', help='Directory to save results')
def evaluate(phrase, model_name, max_tokens, inference_provider, num_examples, output_dir):
    """Evaluate a model with a specific phrase on the GPQA benchmark."""
    os.makedirs(output_dir, exist_ok=True)
    if num_examples == 0:
        num_examples = None
    gpqa_bs = GPQAEvalBS(num_examples=num_examples, n_repeats=1, bs_phrase=phrase)
    res = gpqa_bs(ChatCompletionSamplerHF(
        model=model_name,
        inference_provider=inference_provider,
        max_tokens=max_tokens
    ))
    
    datetime_now = datetime.now()

    timestamp = datetime_now.strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name.replace('/', '_')
    filename = f"{safe_model_name}_{timestamp}.json"
    output_path = os.path.join(output_dir, filename)
    res_dict = res.__dict__
    res_dict["phrase"] = phrase
    
    with open(output_path, 'w') as f:
        json.dump(res_dict, f)
    
    click.echo(f"Evaluation completed. Results saved to {output_path}")

if __name__ == '__main__':
    evaluate()