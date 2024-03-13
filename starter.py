import sagemaker
from sagemaker.pytorch import PyTorch

estimator = PyTorch(entry_point='train.py',
                    source_dir='.',
                    role='arn:aws:iam::xxx:role/xxxxx',
                    framework_version='1.12.1',  
                    py_version='py38',
                    instance_count=1,
                    instance_type='ml.m5.large', 
                    hyperparameters={
                        'epochs': 5,
                        'batch_size': 2048,
                        'learning_rate': 0.01,
                    },
                    output_path='s3://nnv1model/model_output'
                    )
data_channels = {
    'train': 's3://nnv1model/train_data'
}
estimator.fit(data_channels)

