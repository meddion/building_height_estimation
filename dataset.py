import os
import json
import torchvision
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import random

DEBUG = False

# Expects mask to be 0 or 1
def random_colour_masks(image):
  colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0,10)]
  coloured_mask = np.stack([r, g, b], axis=2)
  return coloured_mask

def show_segmented_img(img, mask, boxes, rect_th=3):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_mask = random_colour_masks(mask)
    img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)

    for x, y, w, h in boxes:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=rect_th)

    plt.figure(figsize=(20,30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()


DEFAULT_CLASS_DIVIDERS = [6.0, 9.0, 18.0, 54]
# Plus one for background and final class for buildings taller than 54m.
NUMBER_OF_CLASSES = len(DEFAULT_CLASS_DIVIDERS) + 2
def get_height_class(height: int) -> int:
    assert height > 0, "Height must be positive"

    for i, divider in enumerate(DEFAULT_CLASS_DIVIDERS):
        if height <= divider:
            return i + 1
    
    return len(DEFAULT_CLASS_DIVIDERS) + 1

class BuildingDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir: str, annotations_dir: str, masks_dir: str, transforms = None) -> None:
        super().__init__()
        self.image_dir = images_dir
        self.annotations_dir = annotations_dir
        self.masks_dir = masks_dir

        self.transforms = transforms

        # image_masks_set = set(os.listdir(masks_dir))
        # self.image_names = sorted(filter(lambda img: img in image_masks_set , os.listdir(images_dir)))
        self.image_names = sorted(os.listdir(images_dir))
        assert len(self.image_names) > 0, "No images found in the dataset path"
        assert len(self.image_names) == len(os.listdir(annotations_dir)) == len(os.listdir(masks_dir)), "Number of images and annotations must match"
        if DEBUG:
            print("DEBUG: number of masks", len(self.image_names))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_handle, img_type = self.image_names[idx].split(".", 1)

        total_bbox_area = 0
        bboxes = []
        building_heights = []
        with open(os.path.join(self.annotations_dir, img_handle+".json")) as f:
            annot = json.load(f)
            for shape in annot["shapes"]:
                points = np.array(shape["points"], dtype=np.int32)
                x, y, w, h = cv2.boundingRect(points)
                total_bbox_area += w*h
                bboxes.append((x, y, w, h))
                building_heights.append(shape["group_id"])

        img = read_image(os.path.join(self.image_dir, self.image_names[idx]))
        img = tv_tensors.Image(img)

        mask = read_image(os.path.join(self.masks_dir, img_handle+"."+img_type))
        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[1:]
        # Split the color-encoded mask into a set of binary masks
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        bboxes = torchvision.ops.boxes.box_convert(torch.tensor(bboxes), in_fmt="xywh", out_fmt="xyxy")
        bboxes = tv_tensors.BoundingBoxes(bboxes, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=F.get_size(img))

        # TODO: make it work with pytorch tensors 
        if DEBUG:
            cv2_img = cv2.imread(os.path.join(self.image_dir, self.image_names[idx]))
            cv2_mask = cv2.imread(os.path.join(self.masks_dir, img_handle+"."+img_type), cv2.IMREAD_UNCHANGED)
            cv2_mask[cv2_mask == 128] = 1
            show_segmented_img(cv2_img, mask, bboxes)

        target = {
            "boxes": bboxes,
            "labels": torch.tensor(list(map(get_height_class, building_heights)), dtype=torch.int64), # 1 - building, 0 - background
            "masks": tv_tensors.Mask(masks),
            # Not used by faster RCNN.
            "building_heights": torch.tensor(building_heights),
            "image_id": idx,
            "area": total_bbox_area,
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    """
    TODO: 
    Additionally, if you want to use aspect ratio grouping during training (so that each batch only contains images with similar aspect ratios), 
    then it is recommended to also implement a get_height_and_width method, which returns the height and the width of the image.
    If this method is not provided, we query all elements of the dataset via __getitem__ , which loads the image in memory and is slower than if a custom method is provided.
    """

