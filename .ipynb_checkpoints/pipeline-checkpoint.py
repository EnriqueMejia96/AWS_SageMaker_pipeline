import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker import get_execution_role

# Setup
sagemaker_session = sagemaker.Session()
role = get_execution_role()  # Ensure your SageMaker role has necessary permissions

# Parameters
input_data = ParameterString(
    name="InputDataUrl",
    default_value="s3://dmc-esp-1/ml-project/dataset/data_iris.csv",  # Replace with your dataset URL
)

# Processing step for data preprocessing
sklearn_processor = SKLearnProcessor(
    framework_version="0.23-1",
    role=role,
    instance_type="ml.m5.xlarge",
    instance_count=1
)

preprocess_step = ProcessingStep(
    name="PreprocessData",
    processor=sklearn_processor,
    inputs=[sagemaker.processing.ProcessingInput(
        source=input_data,
        destination="/opt/ml/processing/input",
    )],
    outputs=[
        ProcessingOutput(
            output_name="train",
            source="processing/train_data.csv",
            destination=f"s3://dmc-esp-1/ml-project/preprocess/train"
        ),
        ProcessingOutput(
            output_name="test",
            source="processing/test_data.csvt",
            destination=f"s3://dmc-esp-1/ml-project/preprocess/test"
        ),
        ProcessingOutput(
            output_name="scaler",
            source="processing/scaler.joblib",
            destination=f"s3://dmc-esp-1/ml-project/preprocess/scaler"
        )
    ],
    code="s3://dmc-esp-1/ml-project/code/preprocess.py"  # Update with your S3 path to preprocess.py
)



# Define the pipeline
pipeline = Pipeline(
    name="IrisDataPreprocessingPipeline",  # Name your pipeline
    parameters=[
        input_data,  # Add any additional parameters here
    ],
    steps=[preprocess_step],  # Add any additional steps here
    sagemaker_session=sagemaker_session,
)

# Upsert (create or update) the pipeline
pipeline.upsert(role_arn=role)

# Start the pipeline execution
execution = pipeline.start()

# Optionally, print the execution ARN
print(execution.arn)
