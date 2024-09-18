"""
This script downloads a subset of images from the COCO dataset.

Necessary Files:
- annotations for training images
- annotations for validation images
- annotations for test images

Source for COCO dataset:
https://cocodataset.org/#download

"""

import os
import random
import requests
from pycocotools.coco import COCO
from tqdm import tqdm

# Paths to annotation files
train_ann_file = '../COCO/annotations/captions_train2017.json'
val_ann_file = '../COCO/annotations/captions_val2017.json'
test_ann_file = '../COCO/annotations/image_info_test2017.json'

# Load annotations
coco_train = COCO(train_ann_file)
coco_val = COCO(val_ann_file)
coco_test = COCO(test_ann_file)


# Function to download images
def download_images(coco, img_ids, save_dir):
    print(f"downloading images of {save_dir.split('/')[1]}...")
    for img_id in tqdm(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_url = img_info['coco_url']
        img_data = requests.get(img_url).content
        # Ensure the directory exists
        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/{img_info['file_name']}", 'wb') as handler:
            handler.write(img_data)


# Select a part of images (if indicated)
def select_subset(coco, percentage=1.0):
    img_ids = coco.getImgIds()
    subset_size = int(len(img_ids) * percentage)
    return random.sample(img_ids, subset_size)


# Select and download subsets
train_img_ids = select_subset(coco_train, 0.1)
val_img_ids = select_subset(coco_val, 0.1)
test_img_ids = select_subset(coco_test, 0.1)

download_images(coco_train, train_img_ids, '../COCO/train_subset')
download_images(coco_val, val_img_ids, '../COCO/val_subset')
download_images(coco_test, test_img_ids, '../COCO/test_subset')

print('Done')
