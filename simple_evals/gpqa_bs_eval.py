"""
GPQA: A Graduate-Level Google-Proof Q&A Benchmark
David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, Samuel R. Bowman
https://arxiv.org/abs/2311.12022

Copied from https://github.com/openai/simple-evals/tree/3ec4e9b5ae3931a1858580e2fd3ce80c7fcbe1d9 and changed only correct answer and parsing to BS-fy this benchmark

"""

import random
import re

import pandas

from . import common
from .common import ANSWER_PATTERN_MULTICHOICE, HTML_JINJA, format_multichoice_question, ANSWER_PATTERN_MULTICHOICE_BS
from .types import Eval, EvalResult, MessageList, SamplerBase, SingleEvalResult


class GPQAEvalBS(Eval):
    def __init__(
        self,
        n_repeats: int = 4,
        variant: str = "diamond",
        num_examples: int | None = None,  # restrict to a subset of the data for debugging
        bs_phrase: str = "This is a sample answer",
    ):
        df = pandas.read_csv(
            f"https://openaipublic.blob.core.windows.net/simple-evals/gpqa_{variant}.csv"
        )
        examples = [row.to_dict() for _, row in df.iterrows()]
        rng = random.Random(0)
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported for num_examples = None"
            examples = rng.sample(examples, num_examples)
        examples = examples * n_repeats
        examples = [example | {"permutation": rng.sample(range(4), 4)} for example in examples]
        self.examples = examples
        self.n_repeats = n_repeats
        self.bs_phrase = bs_phrase

    def __call__(self, sampler: SamplerBase, return_messages = None) -> EvalResult:
        def fn(row: dict, return_messages = return_messages):
            real_correct_answer = row["Correct Answer"]
            real_question = row["Question"]
            row["Question"] = row["Question"].replace(row["Correct Answer"], self.bs_phrase)
            row["Correct Answer"] = self.bs_phrase
            choices = [
                row["Correct Answer"],
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]
            choices = [choices[i] for i in row["permutation"]]
            correct_index = choices.index(row["Correct Answer"])
            correct_answer = "ABCD"[correct_index]
            choices_dict = dict(
                A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=row["Question"]
            )
            prompt_messages = [
                sampler._pack_message(
                    content=format_multichoice_question(choices_dict), role="user"
                )
            ]
            if return_messages:
                return prompt_messages, correct_answer, real_correct_answer
            sampler_response = sampler(prompt_messages)
            response_text = sampler_response.response_text
            actual_queried_prompt_messages = sampler_response.actual_queried_message_list
            match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
            extracted_answer = match.group(1) if match else None
            score = 1.0 if extracted_answer == correct_answer else 0.0
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=actual_queried_prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=real_correct_answer,
                extracted_answer=extracted_answer,
            )
            convo = actual_queried_prompt_messages + [dict(content=response_text, role="assistant")]
            row["Correct Answer"] = real_correct_answer
            row["Question"] = real_question
            return SingleEvalResult(
                html=html, score=score, convo=convo, metrics={"chars": len(response_text)}
            )
        results = common.map_with_progress(fn, self.examples)
        if return_messages:
            return list(zip(*results))
        return common.aggregate_results(results)
