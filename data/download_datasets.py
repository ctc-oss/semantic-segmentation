import boto3
import os
import shutil

s3_bucket = "spacenet-dataset"
# s3_base_folder = "Hosted-Datasets/Urban_3D_Challenge"
s3_base_folder = "spacenet/SN6_buildings"
train_dataset_rgb = "train/AOI_11_Rotterdam/PS-RGB/"
train_dataset_sar = "train/AOI_11_Rotterdam/SAR-Intensity/"
# test_dataset_rgb = "test_public/"
test_dataset_sar = "test_public/AOI_11_Rotterdam/SAR-Intensity/"
# train_dataset = "01-Provisional_Train/Inputs"
# test_dataset = "02-Provisional_Test/Inputs"

s3_resource = boto3.resource('s3')
bucket = s3_resource.Bucket(s3_bucket)


def download_dataset(s3_folder_name, local_folder):
    print(f"Downloading dataset {s3_folder_name}...")
    for obj in bucket.objects.filter(Prefix=s3_folder_name):
        if not os.path.exists(os.path.dirname(obj.key)):
            os.makedirs(os.path.dirname(obj.key))
        bucket.download_file(obj.key, obj.key)
    shutil.move(s3_folder_name, local_folder, copy_function=shutil.copy2)
    shutil.rmtree(s3_base_folder.split("/")[0])
    print("Download Completed!\n")


if __name__ == '__main__':
    download_dataset(f"{s3_base_folder}/{train_dataset_rgb}", "data/train/rgb")
    download_dataset(f"{s3_base_folder}/{train_dataset_sar}", "data/train/sar")
    download_dataset(f"{s3_base_folder}/{test_dataset_sar}", "data/test/sar")
