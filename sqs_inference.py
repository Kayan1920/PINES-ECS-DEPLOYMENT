import os
import boto3
import json
from transformers import pipeline
import torch

queue_url = os.environ.get("QUEUE_URL")

sqs = boto3.client("sqs")

device = 0 if torch.cuda.is_available() else -1

model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device,
    max_length=512,
    truncation=True,
)

while True:
    response = sqs.receive_message(
        QueueUrl=queue_url,
        MaxNumberOfMessages=10,
        WaitTimeSeconds=20,
    )

    messages = response.get("Messages", [])

    for message in messages:
        body = json.loads(message["Body"])
        notes = body["notes"]

        results = model(notes)

        print(results)

        sqs.delete_message(
            QueueUrl=queue_url,
            ReceiptHandle=message["ReceiptHandle"],
        )