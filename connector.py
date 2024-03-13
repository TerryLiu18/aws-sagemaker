import boto3
from botocore.exceptions import NoCredentialsError

# Initialize a boto3 client for S3

s3_client = boto3.client(
    's3',
    aws_access_key_id='xxxxx',
    aws_secret_access_key='xxxxxxx',
    region_name='xxxxx'
)

def upload_to_s3(local_file, bucket, s3_file):
    try:
        s3_client.upload_file(local_file, bucket, s3_file)
        print(f"File {local_file} uploaded to {bucket}/{s3_file}")
    except FileNotFoundError:
        print(f"The file {local_file} was not found")
    except NoCredentialsError:
        print("Credentials not available")

# Specify your file and target S3 location
local_file = 'train_label.npy'
bucket_name = 'nnv1model'  # Replace with your actual bucket name
s3_file_name = 'train_label.npy'  # Name for the file in S3

# Upload the file
upload_to_s3(local_file, bucket_name, s3_file_name)

