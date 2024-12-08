import os
import json
import sys
import boto3

prompt = """
    You are a smart assistant so please let me know what is machine learning in the smartest way?
"""

bedrock = boto3.client("bedrock-runtime")

payload = {
    "prompt": "[INST]" + prompt + "[/INST]",
    "max_gen_len": 512,
    "temperature": 0.3,
    "top_p": 0.9,
}

body = json.dumps(payload)
model_id = "meta.llama3-8b-instruct-v1:0"

response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)

response_body = json.loads(response.get("body").read())
response_text = response_body["generation"]

print(response_text)