import click
import json
import os
import dotenv
from datetime import datetime
import llms.util as util
import llms.llms as llms
import re
from data.data import manually_created_tasks
import pandas as pd
dotenv.load_dotenv()

judge_prompt = """You are given task:
<user_request>
{task}
</user_request> 
And model response: 
<model_response>
{model_response}
</model_response>
Does the model response unambiguously state that the request is impossible? If so, answer "yes". If not, answer "no". Do not provide any other information or explanation."""

MODEL_KWARGS = {
    "o4-mini": {"reasoning": {
    "effort": "high"}, "max_completion_tokens": 1024*32},
    "o3-mini": {"reasoning": {
    "effort": "high"}, "max_completion_tokens": 1024*32},
    "gpt-4o-mini": {"max_output_tokens": 32000},
    "claude-sonnet-4-20250514": {"max_tokens": 1024*32},
    "claude-3-7-sonnet-20250219": {"max_tokens": 1024*32},
}

HF_KWARGS = {
    "provider": "nebius",
    "bill_to": "Le-G-AgentX",
}

JUDGE_LLM_NAME = "deepseek-ai/DeepSeek-V3-0324"

@click.command()
@click.option('--system-prompt', required=True, help='The phrase to test')
@click.option('--model-name', required=True, help='Model name to evaluate')
@click.option('--num-examples', type=click.INT, default=None, help='Number of examples to evaluate (None for all)')
@click.option('--max-tokens', default=4096, help='Max number of tokens to use')
@click.option('--output-dir', default='logs/bsbench/responses', help='Directory to save results')
def evaluate(system_prompt, model_name, num_examples, max_tokens, output_dir):
    target_llm = llms.ModelGPT(model_name) if 'gpt' in model_name or any([i in model_name for i in 'o1|o3|o4'.split("|")]) else llms.ModelClaude(model_name) if 'claude' in model_name else llms.ModelHF(model_name, **HF_KWARGS) 

    df = pd.DataFrame(manually_created_tasks, columns=["index", "task", "subtype", "comment"])
    prompts = df['task'].tolist()
    if num_examples:
        prompts = prompts[:num_examples]
    msg_data = [[{"role": "system", "content": system_prompt}, {"role": "user", "content": p}] if system_prompt else  [{"role": "user", "content": p}] for p in prompts]
    # print(msg_data[0])
    model_kwargs = MODEL_KWARGS.get(model_name, {})
    if not model_kwargs:
        model_kwargs = {"max_tokens": 1024*32}
    responses = target_llm.get_responses(
        msgs=msg_data,
        **model_kwargs,
    )
    # print(responses[0])
    try:
        jdg_prompts = [judge_prompt.format(task=m[-1]['content'], model_response = r['choices'][0]['message']['content']) for m, r in zip(msg_data, responses)] 
    except:
        jdg_prompts = [judge_prompt.format(task=m[-1]['content'], model_response = r['content']) for m, r in zip(msg_data, responses)] 
        
    # print(jdg_prompts[0])
    judge_llm = llms.ModelHF(model_name=JUDGE_LLM_NAME, **HF_KWARGS) 
    judgements = judge_llm.get_responses(
        msgs=[[{"role": "user", "content": j}] for j in jdg_prompts],
    )
    res_dict = {
        "msg_data": msg_data,
        "responses": responses,
        "judgements": judgements,
    }
    res_dict['system_prompt'] = system_prompt
    datetime_now = datetime.now()

    timestamp = datetime_now.strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name.replace('/', '_')
    filename = f"{safe_model_name}_{timestamp}.json"
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w') as f:
        json.dump(res_dict, f)
    
    click.echo(f"Evaluation completed. Results saved to {output_path}")

if __name__ == '__main__':
    evaluate()