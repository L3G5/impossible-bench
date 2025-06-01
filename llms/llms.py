import os
import openai
import anthropic
import time
import json
import uuid
import llms.util as util
from tqdm import tqdm
from huggingface_hub import InferenceClient
import logging

class ModelGPT:
    def __init__(self, model_name, env = "OPENAI_API_KEY", batch_completed_status = "completed"):
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=os.getenv(env))
        self.batch_completed_status = batch_completed_status


    def create_batch_file(self, msg_list, input_file_nm, **kwargs):
        """
        Create a JSONL file for the batch API input.
        """
        batch_input = []
        for msg in msg_list:
            body = {
                "model": self.model_name,
                "messages": msg,
            }
            for k in ["temperature", "seed"]:
                v = kwargs.get(k, None)
                if v:
                    body[k] = v
            if self.model_name == "o1" or self.model_name == "o3-mini":
                for k in ["max_tokens"]:
                    v = kwargs.get(k, None)
                    if v and v < 150:
                        body['reasoning_effort'] = "low"
                    elif v and v < 1500:
                        body['reasoning_effort'] = "medium"
                    elif v and v >= 1500:
                        body['reasoning_effort'] = "high"
            elif self.model_name == "o1-mini":
                for k in ["max_tokens"]:
                    v = kwargs.get(k, None)
                    if v:
                        body["max_completion_tokens"] = v
            else:
                for k in ["max_tokens"]:
                    v = kwargs.get(k, None)
                    if v:
                        body[k] = v
            request = {
                "custom_id": str(uuid.uuid4()),
                "method": "POST", 
                "url": "/v1/chat/completions", 
                "body": body
            }
            batch_input.append(request)

        # Save the batch input to a JSONL file
        with open(input_file_nm, "w") as f:
            for request in batch_input:
                f.write(json.dumps(request) + "\n")
            logging.info(request)
        batch_input_file = self.client.files.create(
            file=open(input_file_nm, "rb"),
            purpose="batch"
        )
        return batch_input_file

    def submit_batch(self, batch_input_file):
        """
        Submit the batch job to OpenAI.
        """
        batch = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        # logging.info(batch)
        return batch.id

    def check_batch_status(self, batch_id):
        """
        Check the status of the batch job.
        """
        batch = self.client.batches.retrieve(batch_id)
        logging.info(f"{batch.status=}, {batch.request_counts=}")
        return batch.status

    def download_batch_results(self, batch_id, output_file):
        """
        Download the results of the batch job.
        """
        batch = self.client.batches.retrieve(batch_id)
        # print(batch)
        if batch.status == self.batch_completed_status:
            file_response = self.client.files.content(batch.output_file_id)
            with open(output_file, "w") as f:
                f.write(file_response.text)
            return True
        return False

    def get_responses(self, msgs, **kwargs):
        """
        Get responses for a batch of prompts using the OpenAI Batch API.
        """
        input_file_nm = "batch_input.jsonl"
        input_file = self.create_batch_file(msgs, input_file_nm, **kwargs)
        
        batch_id = self.submit_batch(input_file)
        print(f"Batch submitted. Batch ID: {batch_id}")
        
        while True:
            status = self.check_batch_status(batch_id)
            # print(f"Batch status: {status}")
            if status == self.batch_completed_status:
                break
            time.sleep(60)  # Wait 1 minute before checking again
        
        output_file_nm = "batch_output.jsonl"
        if self.download_batch_results(batch_id, output_file_nm):
            df = util.jsonl_to_df(input_file_nm).merge(util.jsonl_to_df(output_file_nm), how = 'left', on = 'custom_id')
            return [self.parse_response(r) for r in df['response'].tolist()]
        else:
            print("Batch processing failed.")
            return [""] * len(msgs)

    def parse_response(self, response):
        return util.extract_data(response['body'])

class ModelClaude:
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def create_batch_file(self, msg_list, input_file_nm, **kwargs):
        """
        Create a JSONL file for the batch API input.
        """
        batch_input = []
        for msg in msg_list:
            body = {
                "model": self.model_name,
                "messages": msg,
            }
            for k in ["max_tokens", "temperature", "seed"]:
                v = kwargs.get(k, None)
                if v:
                    body[k] = v
            if self.model_name == "claude-3-7-sonnet-20250219":
                body["thinking"] = {
                    "type": "disabled",
                }

            request = {
                "custom_id": str(uuid.uuid4()),
                "params": body
            }
            batch_input.append(request)
        
        # Save the batch input to a JSONL file
        with open(input_file_nm, "w") as f:
            for request in batch_input:
                f.write(json.dumps(request) + "\n")

        batch_input_file = self.client.messages.batches.create(
            requests=batch_input,
        )
        # logging.info(batch_input_file)
        return batch_input_file.id

    def check_batch_status(self, batch_id):
        """
        Check the status of the batch job.
        """
        batch = self.client.messages.batches.retrieve(batch_id)
        logging.info(f"{batch.processing_status=} {batch.request_counts.to_dict()=}")
        return batch.processing_status

    def download_batch_results(self, batch_id, output_file):
        """
        Download the results of the batch job.
        """
        batch = self.client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            file_response = self.client.messages.batches.results(batch_id)
            with open(output_file, "w") as f:
                for line in file_response:
                    f.write(json.dumps(line.to_dict())+'\n')
            return True
        return False

    def get_responses(self, msgs, **kwargs):
        """
        Get responses for a batch of prompts using the Anthropic Batch API.
        """
        input_file_nm = "batch_input_a.jsonl"
        batch_id = self.create_batch_file(msgs, input_file_nm, **kwargs)
        
        print(f"Batch submitted. Batch ID: {batch_id}")
        
        while True:
            status = self.check_batch_status(batch_id)
            # print(f"Batch status: {status}")
            if status == "ended":
                break
            time.sleep(60)  # Wait 1 minute before checking again
        
        output_file_nm = "batch_output_a.jsonl"
        if self.download_batch_results(batch_id, output_file_nm):
            df = util.jsonl_to_df(input_file_nm).merge(util.jsonl_to_df(output_file_nm), how = 'left', on = 'custom_id')
            return [self.parse_response(r['message']) for r in df['result'].tolist()]
        else:
            print("Batch processing failed.")
            return [""] * len(msgs)
        
    def parse_response(self, response):
        parsed = {
            'created': int(time.time()),
            'content': response['content'][0]["text"],
            'model': response['model'],
            'finish_reason': response['stop_reason'],
            'usage': response['usage'],
        }
        return parsed

class ModelHF:
    def __init__(self, model_name: str, provider: str, bill_to: str):
        self.model_name = model_name
        self.client = InferenceClient(
            provider=provider,
            api_key=os.environ["HUGGINGFACE_API_KEY"],
            bill_to=bill_to
        )
        
    def get_responses(self, msgs, **kwargs):
        """
        Get responses for a batch of prompts using the Anthropic Batch API.
        """
        return [self.client.chat.completions.create(
            model=self.model_name,
            messages=msg,
            **kwargs
        ) for msg in msgs]