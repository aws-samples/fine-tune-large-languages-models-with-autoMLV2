import boto3
import os
from botocore.exceptions import ClientError
from datetime import datetime
import json

sagemaker_client = boto3.client("sagemaker")
s3_client = boto3.client("s3")

def lambda_handler(event, context):
    # Obtain the candidtate inference container definitions
    autopilot_job = sagemaker_client.describe_auto_ml_job_v2(
        AutoMLJobName=event["AutopilotJobName"]
    )
    best_candidate = autopilot_job['BestCandidate']
    
    best_candidate_name = best_candidate['CandidateName']
    
    metrics_report_s3_path= event["MetricsReportS3Path"]
    
    model_metrics_report = autopilot_job['BestCandidate']['CandidateProperties']['CandidateMetrics']
    
    # Create a JSON file with the metrics report
    metrics_report_file = '/tmp/metrics_report.json'
    with open(metrics_report_file, 'w') as f:
        json.dump(model_metrics_report, f)
        
    # Upload the JSON file to the specified S3 path
    s3_bucket, s3_key = metrics_report_s3_path.replace("s3://", "").split('/', 1)
    try:
        s3_client.upload_file(metrics_report_file, s3_bucket, s3_key)
        print(f"Metrics report successfully uploaded to {metrics_report_s3_path}")
    except ClientError as e:
        print(f"Error uploading metrics report to S3: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e)
            })
        }    
    
    # create sagemaker model
    model_name = f"autopilot-model-"+ datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    response = sagemaker_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            'Image': autopilot_job["BestCandidate"]["InferenceContainers"][0].pop("Image"),
            'ModelDataUrl': autopilot_job["BestCandidate"]["InferenceContainers"][0].pop("ModelDataUrl"),
            'ImageConfig': {
                'RepositoryAccessMode': 'Platform',
                },
            'Environment': {"HUGGINGFACE_HUB_CACHE": "/tmp", "TRANSFORMERS_CACHE": "/tmp", "HF_MODEL_ID": "/opt/ml/model"}
        },
        ExecutionRoleArn=event["AutopilotExecutionRoleArn"]
    )
    
    model_arn = response["ModelArn"]
    
    endpoint_name = f"ep-{model_name}-automl"
    endpoint_config_name = f"{model_name}-endpoint-config"
    endpoint_configuration = sagemaker_client.create_endpoint_config(
            EndpointConfigName = endpoint_config_name,
            ProductionVariants=[
            {
                'VariantName': "Variant-1",
                'ModelName': model_name,
                'InstanceType': "ml.g5.12xlarge",
                'InitialInstanceCount': 1,
            }
            ],
        )
    response = sagemaker_client.create_endpoint(
                EndpointName=endpoint_name, 
                EndpointConfigName=endpoint_config_name
    )
    endpoint_arn = response["EndpointArn"]   
    
    return {
        "statusCode": 200,
        "body": json.dumps({
            "model_name_param": model_name,
            "endpoint_name_param": endpoint_name,
            "model_metrics_report_param": model_metrics_report,
        })
    }