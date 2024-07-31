import boto3
import os
import json
import tarfile
from urllib.parse import urlparse
from sagemaker.s3_utils import parse_s3_url
from fmeval.data_loaders.data_config import DataConfig
from fmeval.reporting.eval_output_cells import EvalOutputCell
from fmeval.constants import MIME_TYPE_JSONLINES
from fmeval.eval_algorithms.qa_accuracy import QAAccuracy, QAAccuracyConfig
from fmeval.model_runners.sm_model_runner import SageMakerModelRunner
from dataclasses import dataclass
from datetime import datetime

def evaluate_model(eval_dataset_s3_path, endpoint_name, bucket_name, eval_metrics_output_key):

    s3 = boto3.client("s3")
    boto_session = boto3.session.Session()
    aws_region = boto_session.region_name
    sagemaker_client = boto_session.client("sagemaker")

    bucket, object_key = parse_s3_url(eval_dataset_s3_path)
    print(bucket)
    print(object_key)
    s3.download_file(bucket, object_key, "dataset_evaluation.jsonl")
    
    # Read and print the first line of the downloaded file
    with open("dataset_evaluation.jsonl", 'r') as file:
        first_line = file.readline().strip()
        print("First line of dataset_eval.jsonl:", first_line)
    # SageMaker model runner
    sm_endpoint_name = endpoint_name
    sm_model_runner = SageMakerModelRunner(
        endpoint_name=sm_endpoint_name,
        content_template='{"inputs":  $prompt, "parameters": {"max_new_tokens": 10, "top_p": 0.5, "temperature": 0.5, "do_sample" : false}}',
        output="[0].generated_text",
        custom_attributes="accept_eula=true",
    )

    # Eval algorithm configuration
    config = QAAccuracyConfig("<OR>")

    eval_algo = QAAccuracy(config)

    dataset_config = DataConfig(
       dataset_name="dataset_evaluation",
       dataset_uri="dataset_evaluation.jsonl",
       dataset_mime_type=MIME_TYPE_JSONLINES,
       model_input_location="model_input",
       target_output_location="target_output",
    )
    sm_model_runner_prompt_template = """
        inputs: $model_input
        """

    eval_output_all = []
    eval_output = eval_algo.evaluate(
        model=sm_model_runner,
        dataset_config=dataset_config,
        prompt_template=sm_model_runner_prompt_template,
        save=True,
    )
    eval_output_all.append(eval_output)


    # Custom serialization function
    def serialize_eval_output(eval_output):
        return {
             'dataset_scores': [
                 {
                    score.name: { 
                        "value": score.value
                    }
                 } for score in eval_output.dataset_scores
             ]
        }

    # Convert eval_output_all to a serializable format
    serializable_eval_output_all = [[serialize_eval_output(output) for output in eval_outputs] for eval_outputs in eval_output_all]


    # Write evaluation metrics to a JSON file
    evaluation_file = "evaluation_metrics.json"
    with open(evaluation_file, "w") as f:
        json.dump(serializable_eval_output_all, f, indent=4)

    # Upload the JSON file to S3
    s3.upload_file(evaluation_file, bucket_name, eval_metrics_output_key)
    return eval_metrics_output_key