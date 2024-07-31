# Fine-tune Large Language Models (LLMs) with AutoMLV2

This repository contains a comprehensive example of fine tuning LLMs using SageMaker's AutoMLV2 for automated machine learning. The included Jupyter notebook (`automlv2_finetuning.ipynb`), the .py scripts demonstrate the process of preparing data, configuring an AutoML job for fine-tuning, deploying the model, and evaluating its performance.

## Why Fine-tuning LLMs is Important

Fine-tuning large language models (LLMs) is crucial for tailoring them to specific tasks or domains, enhancing their accuracy and relevance in specialized applications. It improves user experience by personalizing responses and addressing gaps in general models, while also being more resource-efficient than training from scratch. This process allows for continuous adaptation to evolving needs and data, ensuring that models remain effective and up-to-date.

## Getting Started

### Prerequisites

- An AWS account
- SageMaker Studio or SageMaker Notebook Instance
- Basic knowledge of Python

### Setup

1. Clone this repository to your local machine or SageMaker environment

2. Open SageMaker Studio or Notebook Instance and navigate to the cloned repository directory.

3. Open the `automlv2_finetuning.ipynb` notebook.

### Running the Notebook

The notebook is divided into sections for ease of understanding and execution:


1. **Training Pipeline**: A first pipeline to load data, train the model and deploy it.
     - ***Load and split data***: A step to split the dataset and load it to s3. It executes the script `load_split_dataset.py`
     - ***Create and start the autopilot job***: Train the model using AutoMLV2 executing the script `start_autopilot_job.py`
     - ***Check job status***: Check the completion of the training job by executing `check_autopilot_job_status.py`
     - ***Create and deploy the model***: Create the model and deploy it to a realtime sagemaker endpoint inference. Use the file `create_autopilot_model.py`
2. **Inference Pipeline**: A second pipeline to evaluate the deployed model, and register it.
    - ***Prepare evaluation data***: Executes the script `preprocess_evaluation.py` to create the dataset.jsonl file needed to evaluate the model
    - ***Evaluate the model***: In `evaluate_model.py`, we use the library fmeval to evaluate the model.
    - ***Register model***: This step uses the script `register_autopilot_model.py` to register the model in the SageMaker registry.

Follow the instructions within the notebook to execute each cell.

## Data preparation

After loading the "allenai/sciq" dataset and splitting it into training and test subsets with an 90-10 split, the code creates DataFrames for both training and validation sets. It then samples 7,000 rows from the training DataFrame for further processing. The relevant fields for fine-tuning are extracted and transformed into a format suitable for an Autopilot job, where each entry is paired with a descriptive instruction and context. This final fine-tuning dataset is then saved to a CSV file on S3, enabling it to be used for model training and evaluation.

## AutoMLV2 Text Generation Job Config

**Official Documentation**:  [Text Generation Config](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_TextGenerationJobConfig.html)

Below is a summary of the config used for this notebook, along with a description of each config arg.

`epochCount`
* __Description__: Determines how many times the model goes through the entire training dataset.<br>
* __Value '3'__: One epoch means the Llama2 model has been exposed to all 7,000 samples and had a chance to learn from them. You can stick to 3, or increase the number, if the model doesn’t converge with just 3 epochs

`learning_rate`
* __Description__: Controls the step size at which a model's parameters are updated during training. It determines how quickly or slowly the model's parameters are updated during training.<br>
* __Value '0.000001'__: A learning rate of 1e-5 or 2e-5 is a good standard when fine-tuning LLMs like llama2.

`batch_size`
* __Description__: Defines the number of data samples used in each iteration of training. It can affect the convergence speed and memory usage.<br>
* __Value '1'__: Start with 1 to avoid out of memory error

`learning_rate_warmup_steps`
* __Description__: Specifies the number of training steps during which the learning rate gradually increases before reaching its target or maximum value.<br>
* __Value '0'__: Start with a value 0


## SageMaker Real-Time Endpoint Inference

Amazon SageMaker Real-Time Endpoint Inference offers the capability to deliver immediate predictions from deployed machine learning models, crucial for scenarios demanding quick decision-making. When an application sends a request to a SageMaker real-time endpoint, it processes the data on-the-fly and returns the prediction instantly. SageMaker Real-Time Endpoint Inference with fine-tuned LLMs can be used for personalized customer support, real-time language translation, and tailored content generation in applications such as chatbots, virtual assistants, and interactive recommendation systems.

## Model evaluation using fmeval
fmeval is a library to evaluate Large Language Models (LLMs) in order to help select the best LLM for your use case. The library evaluates LLMs for multiple tasks such as open-ended generation, text summarization, question answering, classification. For this code, we use the SageMaker Model Runner taking as input the endpoint created in the training pipeline.

## Additional Resources

- [SageMaker AutoMLV2 Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/automl.html)
- [Fine Tuning Best Practices](https://docs.aws.amazon.com/sagemaker/latest/dg/autopilot-create-experiment-finetune-llms.html)

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.

