import os
import cv2
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import torch
import json
import numpy as np
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import v2 as T
from torchvision.ops.boxes import masks_to_boxes
import logging
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch.utils
from typing import Tuple
import torch.utils.data
import utils
from typing import Callable, List, Type
import fnmatch
from skimage.measure import label
from torch import Tensor

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


class BuildingDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        transforms=None,
    ) -> None:
        super().__init__()

        # TODO: rewrite it with img paths
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

                # # TODO: rm
                # if len(annot["shapes"]) > 100:
                #     raise ValueError(
                #         f"More then 100 buildings (got {len(annot['shapes'])}) in the image {self.image_names[idx]}"
                #     )

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


def image_paths(
    root_dir: str, image_name_filter: Callable[[str], bool] = None
) -> List[Path]:
    """
    Returns a list of image names from the given root directories.
    """
    extensions = ["*.png", "*.jpeg", "*.jpg"]
    image_names = []

    for dirname, _dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if not any(fnmatch.fnmatch(filename, ext) for ext in extensions):
                continue

            if image_name_filter is not None and not image_name_filter(filename):
                continue

            relative_filename = Path(dirname) / filename
            image_names.append(relative_filename)

    return sorted(image_names, key=lambda x: str(x))


class MiyazakiDataset(Dataset):
    """
    Miyazaki dataset has images of buildings and their segmentation masks,
    The masks are stored as grey scale images with 'a' prefix and '.png' extension
    and the corresponding building images have the same name with 'i' prefix and '.jpg' extension.
    For example:
        aKEN0002248.png - is a mask for the building image iKEN0002248.jpg
    """

    def __init__(self, root_dir: str, transforms=None):
        self.image_paths = image_paths(root_dir, lambda x: x.startswith("i"))
        self.transforms = transforms
        self.deleted_images = []

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, dict]:
        if idx >= len(self.image_paths):
            raise StopIteration

        try:
            img_path = self.image_paths[idx]
            mask_path = img_path.with_name(img_path.stem.replace("i", "a") + ".png")

            img_path = str(img_path)
            img = read_image(img_path)
            img = tv_tensors.Image(img)

            # Read with cv2 because label expects numpy array.
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            colored_mask, num_labels = label(
                mask, background=0, return_num=True, connectivity=2
            )
            colored_mask = torch.from_numpy(colored_mask)
            obj_ids = torch.tensor(range(num_labels + 1))
            obj_ids = obj_ids[1:]  # Remove the background
            # Split the color-encoded mask into a set of binary masks
            masks = (colored_mask == obj_ids[:, None, None]).to(dtype=torch.uint8)
            masks = tv_tensors.Mask(masks)

            bboxes = masks_to_boxes(masks)
            bboxes = tv_tensors.BoundingBoxes(
                bboxes,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=F.get_size(img),
            )
            area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

            # If there are no bounding boxes, then there are no masks.
            if bboxes.shape[0] != 0:
                degenerate_boxes = bboxes[:, 2:] <= bboxes[:, :2]
                if degenerate_boxes.any():
                    # Get indicies of degenerate boxes to delete
                    indices_to_remove = torch.unique(torch.where(degenerate_boxes)[0])
                    all_indices = torch.arange(bboxes.shape[0])

                    # Create a boolean mask with the indices to keep
                    keep = torch.isin(all_indices, indices_to_remove, invert=True)

                    bboxes = bboxes[keep]
                    area = area[keep]
                    masks = masks[keep]

                if bboxes.min() < 0:
                    raise ValueError("Bounding box values must be positive")

            # There is only one class on masks, a building.
            labels = torch.ones(masks.shape[0], dtype=torch.int64)
            target = {
                "masks": masks,
                "boxes": bboxes,
                "labels": labels,
                "image_id": idx,
                "image_path": img_path,
                "area": area,
            }

            if self.transforms is not None:
                img, target = self.transforms(img, target)

            return img, target

        except Exception as e:
            logging.warn(
                f"Got error while processing image {self.image_paths[idx]} (idx: {idx}): {e}. Deleting it from the image list and skipping it."
            )
            self.deleted_images.append(self.image_paths[idx])
            del self.image_paths[idx]

            return self.__getitem__(idx)


def get_simple_transform(train: bool) -> T.Compose:
    transforms = []

    transforms.append(T.ToDtype(torch.float, scale=True))

    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))

    transforms.append(T.ToPureTensor())

    return T.Compose(transforms)


def get_transform(train: bool) -> T.Compose:
    transforms = []

    transforms.append(T.ToDtype(torch.float, scale=True))

    if train:
        # Horizontal flip
        transforms.append(T.RandomHorizontalFlip(0.5))

        # Vertical flip
        transforms.append(T.RandomVerticalFlip(0.5))

        # Random rotation
        transforms.append(T.RandomRotation(degrees=15))

        # Random crop and resize
        transforms.append(T.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)))

        # Color jitter
        transforms.append(
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        )

        # Random grayscale
        transforms.append(T.RandomGrayscale(p=0.2))

        # Gaussian noise
        transforms.append(T.Lambda(lambda img: img + torch.randn_like(img) * 0.1))

    # Normalize
    transforms.append(
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )

    transforms.append(T.ToPureTensor())

    return T.Compose(transforms)


def data_loaders(
    dataset_root: str,
    dataset_cls: Type,
    get_transform: Callable[[bool], None] = get_simple_transform,
    train_batch_size: int = 2,
    test_batch_size: int = 1,
    test_split: float = 0.2,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns training and test data loaders.
    """
    dataset = dataset_cls(
        dataset_root,
        transforms=get_transform(train=True),
    )

    dataset_test = dataset_cls(
        dataset_root,
        transforms=get_transform(train=False),
    )

    # Split the dataset in train and test set.
    indices = torch.randperm(len(dataset)).tolist()
    test_split = int(test_split * len(dataset))
    dataset = torch.utils.data.Subset(dataset, indices[:-test_split])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-test_split:])

    logging.info(
        f"Size of training set {len(dataset)}. Sise of test set: {len(dataset_test)}"
    )

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=utils.collate_fn,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=utils.collate_fn,
    )

    return data_loader, data_loader_test


def show_segmentation(img, masks, boxes=None, labels=None, bcolors="red"):
    if img.dtype != torch.uint8:
        transform = T.Compose(
            [
                T.Lambda(lambda x: x * 255.0),  # Scale to [0, 255]
                T.Lambda(lambda x: x.to(torch.uint8)),  # Convert to uint8
            ]
        )
        img = transform(img)

    output_image = draw_segmentation_masks(img, masks.to(torch.bool), alpha=0.8)
    if boxes is not None:
        output_image = draw_bounding_boxes(output_image, boxes, labels, colors=bcolors)

    plt.figure(figsize=(9, 9))
    plt.imshow(output_image.permute(1, 2, 0))


def show_segmentation_v2(
    img,
    pred_masks,
    target_masks,
    pred_boxes=None,
    target_boxes=None,
    pred_labels=None,
    target_labels=None,
    pred_title="Predicted",
    font=None,
    font_size=25,
    pred_colors="red",
    target_colors="blue",
):
    # Draw predicted segmentation masks
    output_image_pred = draw_segmentation_masks(
        img.clone(), pred_masks.to(torch.bool), alpha=0.8
    )
    if pred_boxes is not None:
        output_image_pred = draw_bounding_boxes(
            output_image_pred,
            pred_boxes,
            labels=pred_labels,
            colors=pred_colors,
            font=font,
            font_size=font_size,
        )

    # Draw target segmentation masks
    output_image_target = draw_segmentation_masks(
        img.clone(), target_masks.to(torch.bool), alpha=0.8
    )
    if target_boxes is not None:
        output_image_target = draw_bounding_boxes(
            output_image_target,
            target_boxes,
            labels=target_labels,
            colors=target_colors,
            font=font,
            font_size=font_size,
        )

    # Plotting the images
    fig, axs = plt.subplots(1, 2, figsize=(18, 9))

    axs[0].imshow(output_image_pred.permute(1, 2, 0).cpu().numpy())
    axs[0].set_title(pred_title)
    axs[0].axis("off")

    axs[1].imshow(output_image_target.permute(1, 2, 0).cpu().numpy())
    axs[1].set_title("Target")
    axs[1].axis("off")

    plt.show()
