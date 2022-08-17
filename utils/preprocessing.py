from utils import spacenet_utils
import pandas as pd
import cv2
import os


def get_valid_images_name():
    building_labels = pd.read_csv("data/SN6_Train_AOI_11_Rotterdam_Buildings.csv")

    building_labels_list = building_labels["ImageId"].drop_duplicates().tolist()

    print("{} unique masks".format(len(building_labels_list)))

    masks_directory = "data/cloud/masks/"
    if not os.path.exists(masks_directory):
        os.mkdir(masks_directory)

    # Let's save the names of the valid masks to build our training and validation set
    valid_images_name = []

    for img in building_labels_list:
        try:
            filename = masks_directory + img + ".png"
            mask = spacenet_utils.generate_mask_for_image_and_class(
                (900, 900), img, building_labels
            )
            cv2.imwrite(filename, mask)
            valid_images_name.append(img)
        except:  # noqa: E722
            pass

    return valid_images_name
