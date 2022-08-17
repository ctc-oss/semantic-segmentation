import sagemaker
from sagemaker.amazon.amazon_estimator import get_image_uri
from utils.data import setup_data
from utils.env import role, instance_size

(
    train_data,
    train_channel,
    train_annotation_channel,
    val_channel,
    val_annotation_channel,
) = setup_data()

# print(train_data)

sagemaker_session = sagemaker.Session()

bucket = sagemaker_session.default_bucket()
prefix = "sar/semantic-segmentation"
s3_output_location = "s3://{}/{}/output".format(bucket, prefix)
print("Model data will be saved to: {}".format(s3_output_location))

training_image = get_image_uri(
    sagemaker_session.boto_region_name, "semantic-segmentation", repo_version="latest"
)

print(training_image)

# TRAINING
# Create the sagemaker estimator object.
ss_model = sagemaker.estimator.Estimator(
    training_image,  # the semantic segmentation image defined in the previous cell
    role,  # passing on the role to the training job for S3 access
    train_instance_count=1,  # the number of instances on which to train our model
    train_instance_type=instance_size,  # the type of instance, here we need a GPU so we will use an instance from the P familly. # noqa: E501
    train_volume_size=50,  # the volume size for the training instances
    train_max_run=360000,  # a stop condition after 360000 seconds of the training job run
    output_path=s3_output_location,  # the location in  S3 to store our training artefacts, like the model itself
    sagemaker_session=sagemaker_session,
)  # the sagemaker session


ss_model.set_hyperparameters(
    backbone="resnet-50",  # resnet-50 has less layers than resnet-101 so will train faster. You can experiment with resnet 101 on the full dataset. # noqa: E501
    algorithm="deeplab",  # deeplab gave good outcomes in this example. You can experiment with FCN and PSP
    num_classes=2,  # the building class and the non-building class in this example
    epochs=10,  # for the workshop, we will use a low number of epochs. Feel free to experiment with more epochs on the full dataset # noqa: E501
    learning_rate=0.0001,  # the learning rate was selected after an hyperparameter tuning job
    optimizer="adam",  # adam does well on most problems
    mini_batch_size=16,  # smaller batch size will improve training time, you can experiment with this parameter
    validation_mini_batch_size=16,  # smaller batch size will improve training time, you can experiment with this parameter # noqa: E501
    num_training_samples=len(train_data),
)  # the number of training samples

s3_train_data = "s3://{}/{}/data/{}/".format(bucket, prefix, train_channel)
s3_train_annotation = "s3://{}/{}/data/{}/".format(
    bucket, prefix, train_annotation_channel
)
s3_val_data = "s3://{}/{}/data/{}/".format(bucket, prefix, val_channel)
s3_val_annotation = "s3://{}/{}/data/{}/".format(bucket, prefix, val_annotation_channel)


# Create sagemaker s3_input objects
distribution = "FullyReplicated"

train_data = sagemaker.session.s3_input(
    s3_train_data,
    distribution=distribution,
    content_type="image/jpeg",
    s3_data_type="S3Prefix",
)
train_annotation = sagemaker.session.s3_input(
    s3_train_annotation,
    distribution=distribution,
    content_type="image/png",
    s3_data_type="S3Prefix",
)
val_data = sagemaker.session.s3_input(
    s3_val_data,
    distribution=distribution,
    content_type="image/jpeg",
    s3_data_type="S3Prefix",
)
val_annotation = sagemaker.session.s3_input(
    s3_val_annotation,
    distribution=distribution,
    content_type="image/png",
    s3_data_type="S3Prefix",
)

data_channels = {
    "train": train_data,
    "validation": val_data,
    "train_annotation": train_annotation,
    "validation_annotation": val_annotation,
}

# Train
ss_model.fit(inputs=data_channels)
