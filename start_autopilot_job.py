import sys
from pip._internal import main

# Upgrading boto3 to the newest release to be able to use the latest SageMaker features
main(
    [
        "install",
        "-I",
        "-q",
        "boto3",
        "--target",
        "/tmp/",
        "--no-cache-dir",
        "--disable-pip-version-check",
    ]
)
sys.path.insert(0, "/tmp/")
import boto3

sagemaker_client = boto3.client("sagemaker")


def lambda_handler(event, context):
    sagemaker_client.create_auto_ml_job_v2(
        AutoMLJobName=event["AutopilotJobName"],
        AutoMLJobInputDataConfig=[
            {
                "ChannelType": "training",
                "CompressionType": "None",
                "ContentType": "text/csv;header=present", 
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": event["TrainDatasetS3Path"],
                    }
                }
            }
        ],
        DataSplitConfig={
            'ValidationFraction':0.1
        },
        OutputDataConfig={"S3OutputPath": event["TrainingOutputS3Path"]},
        AutoMLProblemTypeConfig={
            "TextGenerationJobConfig": 
            {
                "BaseModelName": event["BaseModelName"],
                'TextGenerationHyperParameters': 
                    {
                        "epochCount": event["epochCount"], 
                        "learningRate": event["learningRate"], 
                        "batchSize": event["batchSize"], 
                        "learningRateWarmupSteps": event["learningRateWarmupSteps"]
                    },
                'ModelAccessConfig': 
                    {
                        'AcceptEula': True
                    }
            }
        },
        RoleArn=event["AutopilotExecutionRoleArn"],
    )