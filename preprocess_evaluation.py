import boto3
import pandas as pd
from datasets import Dataset
from sagemaker.s3_utils import parse_s3_url

def preprocess_evaluation(eval_dataset_s3_path, validation_dataset_s3_path):
    s3 = boto3.client("s3")
    
    bucket, object_key = parse_s3_url(validation_dataset_s3_path)
    s3.download_file(bucket, object_key, "validation.csv")
    # Load the dataset from the local CSV file
    dataset = pd.read_csv('validation.csv')

    fields = ['question', 'correct_answer', 'support']

    dataset_eval_df = dataset[fields].copy()
    dataset_eval_df_cpy = dataset_eval_df.copy()
    dataset_eval_df_cpy["model_input"] = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n### Instruction:\n"
            + dataset_eval_df_cpy["question"]
            + "\n\n### Input:\n"
            + dataset_eval_df_cpy["support"]
        )
    dataset_eval_df_cpy["target_output"] = dataset_eval_df_cpy["correct_answer"]
    autopilot_fields = ['model_input', 'target_output']
    dataset_eval = Dataset.from_pandas(dataset_eval_df_cpy[autopilot_fields])
    dataset_eval.to_json(eval_dataset_s3_path, index=False)
    return eval_dataset_s3_path