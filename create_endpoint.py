# %%
import boto3
import json
import sagemaker
from sagemaker.pytorch import PyTorchModel
import sagemaker
from sagemaker.predictor import Predictor


# %%
# Specify the role ARN, the S3 path to your model, and the framework version
role = 'arn:aws:iam::xxxxxx:role/xxxx'
model_data = 's3://xxxxxxx'
framework_version = '1.12.1'  # Replace with the PyTorch version you used

# Create a SageMaker PyTorch Model
pytorch_model = PyTorchModel(model_data=model_data,
                             role=role,
                             entry_point='inference.py',
                             framework_version=framework_version,
                             py_version='py38')

# Deploy the model to an endpoint
predictor = pytorch_model.deploy(initial_instance_count=1,
                                 instance_type='ml.m5.large',
                                 endpoint_name='my-pytorch-endpoint')
# %%
# Example: sending data for prediction
sample_data = [float(i) for i in range(1878)] 
data = json.dumps(sample_data)
response = predictor.predict(data)
print(response)
