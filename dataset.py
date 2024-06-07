import os
import cv2
import json
import numpy as np
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.ops.boxes import masks_to_boxes
import torch
import logging
from pathlib import Path


MASK_POSTFIX = "_mask"

NUMBER_OF_CLASSES = 2

# These images were found to have bounding boxes that are too small.
IGNORE_IMG_LIST = [
    "bjnwymswty",
    "dayurbqhuw",
    "kqavrcwooa",
    "seypojukdk",
    "zsmmztbbul",
]


class BuildingDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, transforms=None) -> None:
        super().__init__()

        self.root_dir = Path(root_dir)
        extensions = ["*.png", "*.jpeg", "*.jpg"]
        self.image_names = sorted(
            [
                img.stem + img.suffix
                for ext in extensions
                for img in self.root_dir.glob(ext)
                if MASK_POSTFIX not in img.stem and img.stem not in IGNORE_IMG_LIST
            ]
        )
        assert len(self.image_names) > 0, "No images found in the dataset path"

        self.transforms = transforms

        self._delete_image_names = []

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if idx >= len(self.image_names):
            raise StopIteration

        try:
            img_stem, img_type = self.image_names[idx].split(".", 1)
            img = read_image((self.root_dir / self.image_names[idx]))
            img = tv_tensors.Image(img)

            building_heights = []
            masks = []
            with open(os.path.join(self.root_dir / (img_stem + ".json"))) as f:
                annot = json.load(f)
                for shape in annot["shapes"]:
                    mask = np.zeros(img[0, :, :].shape, dtype=np.uint8)
                    points = np.array(shape["points"], dtype=np.int32)
                    cv2.fillPoly(mask, [points], (1, 0, 0))
                    masks.append(mask)
                    building_heights.append(shape["group_id"])

            masks = np.stack(masks, axis=0)
            masks = torch.from_numpy(masks)
            masks = tv_tensors.Mask(masks)
            # Get bounding box coordinates for each mask
            bboxes = masks_to_boxes(masks)
            area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

            assert not torch.any(area < 1e-6), "Bounding box area must be positive"
            assert bboxes.min() >= 0, "Bounding box values must be positive"
            bboxes = tv_tensors.BoundingBoxes(
                bboxes,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=F.get_size(img),
            )

            # There is only one class on masks, a building.
            # 1 - for building, 0 - for background
            labels = torch.ones(masks.shape[0], dtype=torch.int64)

            assert (
                len(labels) == len(masks) == len(bboxes)
            ), f"{len(labels)} (labels) != {len(masks)} (masks) != {len(bboxes)} (bboxes)"

            target = {
                "boxes": bboxes,  # A bbox around each building.
                "labels": labels,
                "masks": masks,
                "building_heights": torch.tensor(building_heights),
                "image_id": idx,
                "image_name": self.image_names[idx],
                "area": area,
                # TODO: remove this after the eval function no longer depends on this field
                "iscrowd": torch.zeros(len(labels), dtype=torch.int64),
            }

            if self.transforms is not None:
                img, target = self.transforms(img, target)

            return img, target
        except Exception as e:
            logging.warn(
                f"Got error while processing image {self.image_names[idx]} (idx: {idx}): {e}. Deleting it from the image list and skipping it."
            )
            self._delete_image_names.append(self.image_names[idx])
            del self.image_names[idx]

            return self.__getitem__(idx)

    """
    TODO: 
    Additionally, if you want to use aspect ratio grouping during training (so that each batch only contains images with similar aspect ratios), 
    then it is recommended to also implement a get_height_and_width method, which returns the height and the width of the image.
    If this method is not provided, we query all elements of the dataset via __getitem__ , which loads the image in memory and is slower than if a custom method is provided.
    """
