import os
import json
import uuid
import boto3
import torch
from dotenv import load_dotenv
from transformers import pipeline
from .helper_funcs import parse_s3_path, download_s3_model

# Load environment variables
load_dotenv()

QUEUE_URL = os.environ.get("QUEUE_URL")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
MODEL_S3_PATH = os.getenv("MODEL_S3_PATH")
MAX_LENGTH = int(os.getenv("MAX_LENGTH"))

# Download and load model
bucket, prefix = parse_s3_path(MODEL_S3_PATH)
local_model_dir = "/tmp/model"
download_s3_model(bucket, prefix, local_model_dir, AWS_REGION)

if not QUEUE_URL:
    raise ValueError("QUEUE_URL must be defined in .env")

device = 0 if torch.cuda.is_available() else -1

print("Loading model into memory...")
model = pipeline(
    "sentiment-analysis",
    model=local_model_dir,
    device=device,
    max_length=MAX_LENGTH,
    truncation=True,
)
print("Model loaded.")

# -------------------------
# SQS Worker Loop
# -------------------------

sqs = boto3.client("sqs", region_name=AWS_REGION)

while True:
    response = sqs.receive_message(
        QueueUrl=QUEUE_URL,
        MaxNumberOfMessages=10,
        WaitTimeSeconds=20,
    )

    messages = response.get("Messages", [])

    for message in messages:
        body = json.loads(message["Body"])
        request_id = body["request_id"]
        notes = body["notes"]
        response_queue = body["response_queue"]

        results = model.predict(notes)
        print(results)

        response_payload = {
            "request_id": request_id,
            "results": results
        }

        sqs.send_message(
            QueueUrl=response_queue,
            MessageBody=json.dumps(response_payload)
        )

        sqs.delete_message(
            QueueUrl=QUEUE_URL,
            ReceiptHandle=message["ReceiptHandle"],
        )