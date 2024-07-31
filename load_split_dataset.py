import sys
from pip._internal import main
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "sagemaker"])

import boto3
from sagemaker.s3_utils import parse_s3_url
import sagemaker
from sagemaker.s3 import S3Uploader
from datasets import load_dataset
from datasets import Dataset
import pandas as pd
from steps.utils import safe_open_w, write_to_file
import json
import os

def load_split_dataset(train_dataset_s3_path, validation_dataset_s3_path):
    
    # Load and split dataset. Change this to your own dataset
    dataset = load_dataset("allenai/sciq", split="train")
    dataset = dataset.train_test_split(test_size=0.1, shuffle=True)
    dataset_training_df = pd.DataFrame(dataset['train'])
    dataset_validation_df = pd.DataFrame(dataset['test'])

    dataset_training_df = dataset_training_df.sample(n=9500, random_state=42, ignore_index=True)

    # prepare training dataset to fit autopilot job.
    fields = ['question', 'correct_answer', 'support']

    dataset_train_ist_df = dataset_training_df[fields].copy()
    dataset_fine_tune_ist = Dataset.from_pandas(dataset_train_ist_df)

    dataset_fine_tune_ist_cpy= dataset_train_ist_df.copy()
    dataset_fine_tune_ist_cpy["input"] = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n### Instruction:\n"
            + dataset_fine_tune_ist_cpy["question"]
            + "\n\n### Input:\n"
            + dataset_fine_tune_ist_cpy["support"]
        )
    dataset_fine_tune_ist_cpy["output"] = dataset_fine_tune_ist_cpy["correct_answer"]
    autopilot_fields = ['input', 'output']
    dataset_fine_tune = Dataset.from_pandas(dataset_fine_tune_ist_cpy[autopilot_fields])
    dataset_fine_tune.to_csv(train_dataset_s3_path, index=False)

    # save validation data to be processed before evaluation step in the inference pipeline
    dataset_validation = Dataset.from_pandas(dataset_validation_df)
    dataset_validation.to_csv(validation_dataset_s3_path, index=False)
