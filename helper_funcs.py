import os
import boto3
from urllib.parse import urlparse

# -------------------------
# Parse S3 path
# -------------------------
def parse_s3_path(s3_uri):
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    return bucket, prefix


# -------------------------
# Download model from S3
# -------------------------
def download_s3_model(bucket, prefix, local_model_dir, AWS_REGION):
    print("Downloading model from S3...")
    s3 = boto3.client("s3", region_name=AWS_REGION)

    os.makedirs(local_model_dir, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            relative_path = key[len(prefix):].lstrip("/")
            if not relative_path:
                continue

            local_file_path = os.path.join(local_model_dir, relative_path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            print(f"Downloading {key}")
            s3.download_file(bucket, key, local_file_path)

    print("Model download complete.")
