from utils.preprocessing import get_valid_images_name
from utils.env import (
    masks_directory,
    train_annotation_channel,
    val_annotation_channel,
    train_channel,
    val_channel,
    test_channel,
    test_annotation_channel,
)
from sklearn.model_selection import train_test_split
from shutil import copyfile
from osgeo import gdal
import os
import cv2
import numpy as np
import sagemaker
import boto3
import subprocess as sp

sagemaker_session = sagemaker.Session()
role = role = "role_name_with_sagemaker_permissions"
region = sagemaker.session.Session().boto_region_name

bucket = sagemaker_session.default_bucket()
prefix = "sar/semantic-segmentation"

s3_resource = boto3.resource("s3")
s3_bucket = s3_resource.Bucket(bucket)

os.makedirs(masks_directory, exist_ok=True)
os.makedirs(train_annotation_channel, exist_ok=True)
os.makedirs(val_annotation_channel, exist_ok=True)
os.makedirs(train_channel, exist_ok=True)
os.makedirs(val_channel, exist_ok=True)
os.makedirs(test_channel, exist_ok=True)
os.makedirs(test_annotation_channel, exist_ok=True)


def setup_data(full_data_setup=False, copy_to_s3=False):
    valid_images_name = get_valid_images_name()

    # Let's split the data in training set and test set
    train_data, test_data = train_test_split(
        valid_images_name, test_size=0.1, random_state=42
    )

    # We will split once more for get a validation set
    train_data, validation_data = train_test_split(
        train_data, test_size=0.2, random_state=42
    )

    data_directory = "data/train/sar"

    if full_data_setup:
        for filename in os.listdir(data_directory):
            try:
                if filename in train_data or validation_data:
                    raster = gdal.Open("{}/{}".format(data_directory, filename))
                    data = raster.ReadAsArray()
                    data = np.moveaxis(data, 0, -1)
                    img_name = filename[41:-4]
                    if img_name in train_data:
                        cv2.imwrite(
                            "{}/{}.jpg".format(train_channel, img_name), data[:, :, 0]
                        )
                        print("{}/{}.jpg Created!".format(train_channel, img_name))
                    elif img_name in validation_data:
                        cv2.imwrite(
                            "{}/{}.jpg".format(val_channel, img_name), data[:, :, 0]
                        )
                        print("{}/{}.jpg Created!".format(val_channel, img_name))
                    elif img_name in test_data:
                        cv2.imwrite(
                            "{}/{}.jpg".format(test_channel, img_name), data[:, :, 0]
                        )
                        print("{}/{}.jpg Created!".format(test_channel, img_name))
            except:  # noqa: E722
                pass

        for filename in train_data:
            copyfile(
                "{}{}.png".format(masks_directory, filename),
                "{}/{}.png".format(train_annotation_channel, filename),
            )
        for filename in validation_data:
            copyfile(
                "{}{}.png".format(masks_directory, filename),
                "{}/{}.png".format(val_annotation_channel, filename),
            )
        for filename in test_data:
            copyfile(
                "{}{}.png".format(masks_directory, filename),
                "{}/{}.png".format(test_annotation_channel, filename),
            )

    if copy_to_s3:
        for channel in [
            train_channel,
            val_channel,
            train_annotation_channel,
            val_annotation_channel,
        ]:
            # s3_bucket.download_file(obj.key, obj.key)
            sp.run(
                f"aws s3 cp ./{channel}/ s3://{bucket}/{prefix}/data/{channel.split('/')[-1]}/ --recursive --quiet",
                shell=True,
            )
            print(f"Data uploaded to AWS S3 Bucket {bucket} at {prefix}/data/")

    return (
        train_data,
        train_channel,
        train_annotation_channel,
        val_channel,
        val_annotation_channel,
    )
