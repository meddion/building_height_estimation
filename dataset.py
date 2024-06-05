import os
import json
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.ops.boxes import masks_to_boxes
import torch
from pathlib import Path


# Splits the color-encoded mask into a set of binary masks
def convert_to_binary_masks(mask: torch.Tensor) -> torch.Tensor:
    R = mask[0, :, :].long()
    G = mask[1, :, :].long()
    B = mask[2, :, :].long()
    mask = 256**2 * R + 256 * G + B
    obj_ids = mask.unique()
    obj_ids = obj_ids[1:]

    return (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)


MASK_POSTFIX = "_mask"

NUMBER_OF_CLASSES = 2


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
                if MASK_POSTFIX not in img.stem
            ]
        )
        assert len(self.image_names) > 0, "No images found in the dataset path"

        self.transforms = transforms

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_stem, img_type = self.image_names[idx].split(".", 1)

        building_heights = []
        with open(os.path.join(self.root_dir / (img_stem + ".json"))) as f:
            annot = json.load(f)
            for shape in annot["shapes"]:
                building_heights.append(shape["group_id"])

        img = read_image((self.root_dir / self.image_names[idx]))
        img = tv_tensors.Image(img)

        mask = read_image(self.root_dir / (img_stem + MASK_POSTFIX + "." + img_type))
        masks = convert_to_binary_masks(mask)

        # Get bounding box coordinates for each mask
        bboxes = masks_to_boxes(masks)
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])

        # There is only one class on masks, a building.
        # 1 - for building, 0 - for background
        labels = torch.ones(masks.shape[0], dtype=torch.int64)

        assert bboxes.min() >= 0, "Bounding box values must be positive"
        bboxes = tv_tensors.BoundingBoxes(
            bboxes,
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=F.get_size(img),
        )

        assert (
            len(labels) == len(masks) == len(bboxes)
        ), f"{len(labels)} (labels) != {len(masks)} (masks) != {len(bboxes)} (bboxes)"

        target = {
            "boxes": bboxes,  # A bbox around each building.
            "labels": labels,
            "masks": tv_tensors.Mask(masks),
            "building_heights": torch.tensor(building_heights),
            "image_id": idx,
            "area": area,
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
