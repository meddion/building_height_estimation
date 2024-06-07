import torch.utils
import os
from typing import Callable, Tuple
import time
from torch import nn
import torch.utils.data
import torch.nn.functional as F
import logging
import re
from pathlib import Path
import utils
from dataset import data_loaders, NUMBER_OF_CLASSES
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import (
    MaskRCNNPredictor,
)
import torchvision
from engine import train_one_epoch, evaluate
from custom_roi_heads import CustomRoIHeads
from dataclasses import dataclass
from torch.utils.data import DataLoader


logging.basicConfig(level=logging.INFO)


class TwoMLPRegression(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        hiddden_features = in_features // 2
        self.ln1 = nn.Linear(in_features, hiddden_features)
        self.ln2 = nn.Linear(hiddden_features, 1)

    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = self.ln2(x)

        return x


class EnhancedTwoMLPRegression(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        hidden_features = in_features // 2
        self.ln1 = nn.Linear(in_features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.dropout1 = nn.Dropout(p=0.5)

        self.ln2 = nn.Linear(hidden_features, hidden_features // 2)
        self.bn2 = nn.BatchNorm1d(hidden_features // 2)
        self.dropout2 = nn.Dropout(p=0.5)

        self.ln3 = nn.Linear(hidden_features // 2, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.ln1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.ln2(x)))
        x = self.dropout2(x)
        x = self.ln3(x)

        return x


@dataclass
class ModelConfig:
    building_height_pred: nn.Module
    building_height_pred_loss_fn: Callable
    name: str = "default_model"
    num_classes: int = NUMBER_OF_CLASSES
    mask_hidden_layer_size: int = 256


def new_model(cfg: ModelConfig):
    # Load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # Get number of input features for the classifier.
    # This is a size of features that we get from the backbone.
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Add a new regression head to predict building height
    building_height_pred = cfg.building_height_pred(in_features)

    # Replace the pre-trained head with a new one
    box_predictor = FastRCNNPredictor(in_features, cfg.num_classes)
    # Get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    mask_predictor = MaskRCNNPredictor(
        in_features_mask, cfg.mask_hidden_layer_size, cfg.num_classes
    )

    # Copy all of the params passed to a default RoIHeads
    model.roi_heads = CustomRoIHeads(
        building_height_pred,
        sample_equal=True,  # Sample equal number of positive and negative examples for height regression
        loss_fn=cfg.building_height_pred_loss_fn,
        # RoIHeads inputs
        box_roi_pool=model.roi_heads.box_roi_pool,
        box_head=model.roi_heads.box_head,
        box_predictor=box_predictor,
        # Faster R-CNN training
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.5,
        batch_size_per_image=512,  # Batch size of the RoI minibatch, per image
        positive_fraction=0.25,  # Fraction of RoI minibatch that is labeled as positive
        bbox_reg_weights=None,
        # Faster R-CNN inference
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=100,
        # Mask
        mask_head=model.roi_heads.mask_head,
        mask_predictor=mask_predictor,
        mask_roi_pool=model.roi_heads.mask_roi_pool,
    )

    return model


class Checkpoint:
    def __init__(self, root_dir: Path, model_name: str) -> None:
        self.root_dir = root_dir
        self.model_name = model_name

    def load_latest(self) -> Tuple[dict, str]:
        checkpoint_path = self._latest_epoch_checkpoint()

        return torch.load(checkpoint_path), checkpoint_path

    def save(self, epoch: int, model, optimizer, lr_scheduler, latest_loss: float):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "loss": latest_loss,
            },
            self._file_name(epoch),
        )

    def prune_old(self, threshold: int):
        files = self._sorted_checkpoint_files()

        if len(files) < threshold:
            return

        for file in files[:-threshold]:
            os.remove(file)

    def _latest_epoch_checkpoint(self) -> Path:
        files = self._sorted_checkpoint_files()

        if len(files) == 0:
            raise FileNotFoundError

        return files[-1]

    def _sorted_checkpoint_files(self) -> list[Path]:
        def extract_number(file_obj: Path):
            match = re.search(r"\d+", file_obj.stem)
            return int(match.group()) if match else 0

        files = list(self.root_dir.glob(f"{self.model_name}_epoch_*.pt"))

        return sorted(files, key=extract_number)

    def _file_name(self, epoch) -> Path:
        self.root_dir.mkdir(parents=True, exist_ok=True)

        return self.root_dir / f"{self.model_name}_epoch_{epoch}.pt"


def train(
    data_loader: DataLoader,
    data_loader_test: DataLoader,
    model_cfg: ModelConfig,
    num_epochs: int,
    checkpoint_dir: Path,
    checkpoint_prune_threshold: int,
    # Constructors
    new_optimizer: Callable = None,
    new_lr_scheduler: Callable = None,
    print_freq: int = 10,
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = new_model(model_cfg)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]

    if new_optimizer is None:
        # The weight_decay parameter in the optimizer adds a penalty to the loss function
        # based on the L2 norm of the weights, encouraging smaller weights.
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    else:
        optimizer = new_optimizer(params)

    if new_lr_scheduler is None:
        # The learning rate scheduler reduces the learning rate by a factor of 0.1 every 3 epochs.
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=0.1
        )
    else:
        lr_scheduler = new_lr_scheduler(optimizer)

    start_epoch = 0

    checkpoint = Checkpoint(checkpoint_dir, model_cfg.name)

    try:
        checkpoint_dict, checkpoint_path = checkpoint.load_latest()
        model.load_state_dict(checkpoint_dict["model_state_dict"])
        optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint_dict["lr_scheduler_state_dict"])
        start_epoch = checkpoint_dict["epoch"] + 1
        loss = checkpoint_dict["loss"]
        logging.info(
            f"Loaded the {checkpoint_path} checkpoint, starting training at the {start_epoch} epoch with loss {loss}",
        )
    except FileNotFoundError:
        logging.info("No checkpoint found, starting from scratch")

    for epoch in range(start_epoch, num_epochs):
        # Train for one epoch, printing every print_freq
        _, latest_loss = train_one_epoch(
            model,
            optimizer,
            data_loader,
            device,
            epoch,
            num_epochs,
            print_freq=print_freq,
        )

        # Update the learning rate
        lr_scheduler.step()

        # Evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

        logging.debug(f"Saving model parameters on {epoch} with loss {latest_loss}")

        checkpoint.save(epoch, model, optimizer, lr_scheduler, latest_loss)
        checkpoint.prune_old(checkpoint_prune_threshold)


# TODO: impl proper evaluation
@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    for images, targets in metric_logger.log_every(data_loader, 100, header="Test:"):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        model_time = time.time()

        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        evaluator_time = time.time()
        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        logging.debug(f"Results: {res}")
        # TODO: Run evaluator
        evaluator_time = time.time() - evaluator_time

        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    torch.set_num_threads(n_threads)


def test_predict(
    model_cfg: nn.Module,
    checkpoint_path: str,
    data_loader: DataLoader = None,
    img_path=None,
):
    model = new_model(model_cfg)

    checkpoint_dict = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint_dict["model_state_dict"])

    model.eval()

    if img_path is not None:
        img = torchvision.io.read_image(img_path).unsqueeze(0)
        transforms = T.Compose(
            [
                T.ToDtype(torch.float, scale=True),
                T.ToPureTensor(),
            ]
        )
        img = transforms(img)
    else:
        img = next(iter(data_loader))[0]

    return model(img)


if __name__ == "__main__":
    data_loader, data_loader_test = data_loaders(
        "datasets/mlc_training_data/images_annotated",
        train_batch_size=2,
        test_batch_size=1,
        test_split=0.2,
        num_workers=4,
    )

    model_cfg = ModelConfig(
        name="default_model_v2",
        num_classes=NUMBER_OF_CLASSES,
        mask_hidden_layer_size=256,
        building_height_pred=EnhancedTwoMLPRegression,
        building_height_pred_loss_fn=nn.SmoothL1Loss(beta=1 / 9),
    )

    # print(
    #     test_predict(
    #         model_cfg,
    #         "checkpoints/default_model_epoch_1.pt",
    #         data_loader=data_loader,
    #         # "datasets/mlc_training_data/images_annotated/uqpgutrlld.png",
    #     )
    # )

    train(
        data_loader,
        data_loader_test,
        model_cfg=model_cfg,
        num_epochs=10,
        checkpoint_dir=Path("checkpoints"),
        checkpoint_prune_threshold=3,
    )
