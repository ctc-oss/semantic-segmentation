import sagemaker
from sagemaker.amazon.amazon_estimator import get_image_uri
from utils.env import role, saved_model_folder, endpoint_name, instance_size, region
import sys
import boto3

sagemaker_session = sagemaker.Session()

bucket = sagemaker_session.default_bucket()
prefix = "sar/semantic-segmentation"
saved_model_location = (
    f"s3://{bucket}/{prefix}/output/{saved_model_folder}/output/model.tar.gz"
)
sagemaker_session = sagemaker.Session()
training_image = get_image_uri(
    sagemaker_session.boto_region_name, "semantic-segmentation", repo_version="latest"
)


# Delete endpoint
def undeploy():
    sagemaker_client = boto3.client("sagemaker", region_name=region)
    sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)


# Deploy trained model to sagemaker endpoint
def deploy():
    trained_model = sagemaker.model.Model(
        image=training_image,
        model_data=saved_model_location,
        role=role,
    )

    trained_model.deploy(
        initial_instance_count=1,
        instance_type=instance_size,
        endpoint_name=endpoint_name,
    )


if __name__ == "__main__":
    if len(sys.argv) == 0:
        deploy()
    elif sys.argv[0].lower() == "deploy":
        deploy()
    elif sys.argv[0].lower() == "undeploy":
        undeploy()
