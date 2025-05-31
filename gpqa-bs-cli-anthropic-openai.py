import click
import simple_evals.common as common
from simple_evals.common import ANSWER_PATTERN_MULTICHOICE, HTML_JINJA
from simple_evals.gpqa_bs_eval import GPQAEvalBS
from simple_evals.sampler.chat_completion_sampler_hf import ChatCompletionSamplerHF
import json
import os
import dotenv
from datetime import datetime
import llms.util as util
import llms.llms as llms
import re
dotenv.load_dotenv()

MODEL_KWARGS = {
    "o4-mini": {"reasoning": {
    "effort": "high"}, "max_completion_tokens": 1024*32},
    "o3-mini": {"reasoning": {
    "effort": "high"}, "max_completion_tokens": 1024*32},
    "gpt-4o-mini": {"max_output_tokens": 32000},
    "claude-sonnet-4-20250514": {"max_tokens": 1024*32},
    "claude-3-7-sonnet-20250219": {"max_tokens": 1024*32},
}


@click.command()
@click.option('--phrase', required=True, help='The phrase to test')
@click.option('--model-name', required=True, help='Model name to evaluate')
@click.option('--num-examples', type=click.INT, default=None, help='Number of examples to evaluate (None for all)')
@click.option('--output-dir', default='logs/responses', help='Directory to save results')
def evaluate(phrase, model_name, num_examples, output_dir):
    model_class = llms.ModelGPT if 'gpt' in model_name or any([i in model_name for i in 'o1|o3|o4'.split("|")]) else llms.ModelClaude if 'claude' in model_name else ""
    target_llm = model_class(model_name)  

    if num_examples == 0:
        num_examples = None
    gpqa_bs = GPQAEvalBS(num_examples=num_examples, n_repeats=1, bs_phrase=phrase)
    prompt_messages, correct_answers, real_correct_answers = gpqa_bs(ChatCompletionSamplerHF(
        model=model_name,
    ), return_messages=True)

    model_kwargs = MODEL_KWARGS.get(model_name, {})
    responses = target_llm.get_responses(
        msgs=prompt_messages,
        **model_kwargs,
    )
    res_dict = []
    htmls = []
    scores = []
    convos = []
    full_responses = []
    for m, r, ca, rca in zip(prompt_messages, responses, correct_answers, real_correct_answers):
        response_text = r['content']
        match = re.search(ANSWER_PATTERN_MULTICHOICE, r['content'])
        extracted_answer = match.group(1) if match else None
        score = 1.0 if extracted_answer == ca else 0.0
        html = common.jinja_env.from_string(HTML_JINJA).render(
            prompt_messages=m,
            next_message=dict(content=response_text, role="assistant"),
            score=score,
            correct_answer=rca,
            extracted_answer=extracted_answer,
        )
        convo = m + [dict(content=response_text, role="assistant")]
        htmls += [html]
        scores += [score]
        convos += [convo]
        full_responses += [r]

        # res_dict.append({
        #     "html": html,
        #     "score": score,
        #     "convo": convo,
        #     "full_response": r,
        # })
    res_dict = {
        "htmls": htmls,
        "scores": scores,
        "convos": convos,
        "full_responses": full_responses,
    }
    datetime_now = datetime.now()

    timestamp = datetime_now.strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name.replace('/', '_')
    filename = f"{safe_model_name}_{timestamp}.json"
    output_path = os.path.join(output_dir, filename)
    res_dict["phrase"] = phrase
    
    with open(output_path, 'w') as f:
        json.dump(res_dict, f)
    
    click.echo(f"Evaluation completed. Results saved to {output_path}")

if __name__ == '__main__':
    evaluate()