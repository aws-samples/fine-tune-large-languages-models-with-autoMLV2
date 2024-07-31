import boto3
import os
from botocore.exceptions import ClientError
from urllib.parse import urlparse
from datetime import datetime

s3_client = boto3.client("s3")
sagemaker_client = boto3.client("sagemaker")

RANDOM_SUFFIX = "model" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def lambda_handler(event, context):
    # Get the explainability results from the Autopilot job
    autopilot_job = sagemaker_client.describe_auto_ml_job_v2(
        AutoMLJobName=event["AutopilotJobName"]
    )
    
    #Extract metrics from the best candidate
    metrics = autopilot_job["BestCandidate"]["CandidateProperties"]["CandidateMetrics"]

    # Convert metrics to the required format
    metric_list = [
        {
            "Name": metric["StandardMetricName"],
            "Value": metric["Value"],
        }
        for metric in metrics
    ]

    autopilot_job["BestCandidate"]["InferenceContainers"][0].pop("Environment")
    
    
    # Register the model in Model Registry
    model_package_group_name = 'model-autopilot-package-group'+ datetime.now().strftime("%Y-%m-%d-%H-%M-%S")  
    
    model_package_description = 'Model registered from Autopilot job'
    model_package_group_response = sagemaker_client.create_model_package_group(
        ModelPackageGroupName=model_package_group_name,
        ModelPackageGroupDescription=model_package_description
    )
    register_model_response = sagemaker_client.create_model_package(
        ModelPackageGroupName=model_package_group_name,
        ModelApprovalStatus=event["ModelApprovalStatus"],
        InferenceSpecification={
            "Containers": autopilot_job["BestCandidate"]["InferenceContainers"],
            "SupportedContentTypes": ["text/csv"],
            "SupportedResponseMIMETypes": ["text/csv"],
            "SupportedTransformInstanceTypes": [event["InstanceType"]],
            "SupportedRealtimeInferenceInstanceTypes": [event["InstanceType"]],
        },  
        ModelPackageDescription=model_package_description,
        ModelMetrics={
            "ModelQuality": {
                "Statistics": {
                    "ContentType": "application/json",
                    "S3Uri": event["EvalMetricsOutputS3Path"]
                    ,
                },
            }
        },
    )
   