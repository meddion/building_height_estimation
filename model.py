import torch.utils
import os
from typing import Callable, Optional, Type, Dict, List, Tuple
from torch import nn
import torch.utils.data
import torch.nn.functional as F
import logging
import re
from pathlib import Path
from dataset import (
    NUMBER_OF_CLASSES,
    BuildingDataset,
    data_loaders,
    show_segmentation_diffs,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import (
    MaskRCNNPredictor,
)
import torchvision
from custom_roi_heads import CustomRoIHeads
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
import torch
from sklearn.metrics import jaccard_score
from engine import train_one_epoch
from torchvision.models.detection.rpn import AnchorGenerator
from torch.nn import SmoothL1Loss


logging.basicConfig(level=logging.INFO)


class TwoMLPRegression(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        hiddden_features = in_features // 2
        self.ln1 = nn.Linear(in_features, hiddden_features)
        self.ln2 = nn.Linear(hiddden_features, 1)

    def forward(self, x):
        x = x.flatten(start_dim=1)

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
        # TODO: think about it
        x = x.flatten(start_dim=1)

        x = F.relu(self.bn1(self.ln1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.ln2(x)))
        x = self.dropout2(x)
        x = self.ln3(x)

        return x


class SimpleBuildingHeightPred(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.ln1 = nn.Linear(in_features, 1)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        return self.ln1(x)


@dataclass
class ModelConfig:
    name: str = "default_model"
    num_classes: int = NUMBER_OF_CLASSES
    mask_hidden_layer_size: int = 256
    #  ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_backbone_layers])
    trainable_backbone_layers: int = 3
    building_height_pred_class: Optional[Type] = None
    building_height_pred_loss_fn: Optional[Callable] = None
    height_pred_after_roi_pool: bool = False
    sample_equal: bool = False


def new_model(cfg: ModelConfig) -> nn.Module:
    # Load an instance segmentation model pre-trained on COCO
    # anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    # anchor_sizes = ((4,), (8,), (16,), (32,), (64,), (128,))
    anchor_sizes = (
        (4,),
        (8,),
        (16,),
        (32,),
    )
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights="DEFAULT",
        trainable_backbone_layers=cfg.trainable_backbone_layers,
        anchor_generator=anchor_generator,
        rpn_pre_nms_top_n_train=2 * 2000,
        rpn_pre_nms_top_n_test=2 * 1000,
        rpn_post_nms_top_n_train=2 * 2000,
        rpn_post_nms_top_n_test=2 * 1000,
        rpn_batch_size_per_image=2 * 256,
        box_batch_size_per_image=2 * 512,
        box_detections_per_img=2 * 100,
    )
    model.config = cfg

    # Get number of input features for the classifier.
    # This is a size of features that we get from the backbone.

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    box_predictor = FastRCNNPredictor(in_features, cfg.num_classes)

    # Get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    mask_predictor = MaskRCNNPredictor(
        in_features_mask, cfg.mask_hidden_layer_size, cfg.num_classes
    )

    # Create a new regression head to predict building height
    building_height_pred_head = new_height_prediction_head(
        model,
        cfg.building_height_pred_class,
        cfg.height_pred_after_roi_pool,
    )

    if building_height_pred_head is not None:
        assert (
            cfg.building_height_pred_loss_fn is not None
        ), "Loss function must be provided"

    # Copy all of the params passed to a default RoIHeads
    model.roi_heads = CustomRoIHeads(
        building_height_pred_head,
        cfg.building_height_pred_loss_fn,
        cfg.height_pred_after_roi_pool,
        cfg.sample_equal,  # Sample equal number of positive and negative examples for height regression
        # RoIHeads inputs
        box_roi_pool=model.roi_heads.box_roi_pool,
        box_head=model.roi_heads.box_head,
        box_predictor=box_predictor,
        # Faster R-CNN training
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.5,
        batch_size_per_image=2 * 512,  # Batch size of the RoI minibatch, per image
        positive_fraction=0.25,  # Fraction of RoI minibatch that is labeled as positive
        bbox_reg_weights=None,
        # Faster R-CNN inference
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=2 * 100,
        # Mask
        mask_head=model.roi_heads.mask_head,
        mask_predictor=mask_predictor,
        mask_roi_pool=model.roi_heads.mask_roi_pool,
    )

    return model


def new_height_prediction_head(
    model: nn.Module,
    height_pred_class: Type | None,
    pred_after_roi_pool: bool,
) -> nn.Module | None:
    if height_pred_class is None:
        return None

    # Add a new regression head to predict building height
    if pred_after_roi_pool:
        resolution = model.roi_heads.box_roi_pool.output_size[0]
        out_channels = model.backbone.out_channels
        return height_pred_class(out_channels * resolution**2)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    return height_pred_class(in_features)


def set_height_prediction_head(
    model: nn.Module,
    height_pred_class: Type,
    height_pred_loss_fn: Type,
    height_pred_after_roi_pool: bool,
    sample_equal: bool,
):
    assert isinstance(model.roi_heads, CustomRoIHeads), "Model must be CustomRoIHeads"
    assert height_pred_loss_fn is not None, "Loss function must be provided"

    height_predictor = new_height_prediction_head(
        model,
        height_pred_class,
        height_pred_after_roi_pool,
    )

    model.roi_heads.set_parameters(
        height_predictor,
        height_pred_loss_fn,
        height_pred_after_roi_pool,
        sample_equal,
    )

    model.config.building_height_pred_class = height_pred_class
    model.config.building_height_pred_loss_fn = height_pred_loss_fn
    model.config.height_pred_after_roi_pool = height_pred_after_roi_pool
    model.config.sample_equal = sample_equal


def new_pretrained_model(
    new_model_name: str,
    pretrained_checkpoint_path: str,
    # height pred params
    height_pred_class: Type,
    height_pred_loss_fn: Type,
    height_pred_after_roi_pool: bool = False,
    sample_equal: bool = False,
    # torch load params
    strict: bool = True,
    **load_kwargs,
) -> nn.Module:
    """Load a pretrained instance segmentation model from a checkpoint file and set height prediction head."""

    checkpoint_dict = torch.load(pretrained_checkpoint_path, **load_kwargs)

    model_cfg = ModelConfig(
        name=new_model_name,
        trainable_backbone_layers=0,  # Freeze the entire ResNet encoder for fine-tuning
    )

    if checkpoint_dict.get("hyperparameters") is not None:
        loaded_model_cfg = checkpoint_dict["hyperparameters"]["model_cfg"]
        logging.info(f"Loaded model config: {loaded_model_cfg}")
        model_cfg.num_classes = loaded_model_cfg.num_classes
        model_cfg.mask_hidden_layer_size = loaded_model_cfg.mask_hidden_layer_size

    model = new_model(model_cfg)

    model.load_state_dict(checkpoint_dict["model_state_dict"], strict=strict)

    set_height_prediction_head(
        model,
        height_pred_class,
        height_pred_loss_fn,
        height_pred_after_roi_pool,
        sample_equal,
    )

    model.config.building_height_pred_class = height_pred_class
    model.config.building_height_pred_loss_fn = height_pred_loss_fn
    model.config.height_pred_after_roi_pool = height_pred_after_roi_pool
    model.config.sample_equal = sample_equal

    return model


def load_fine_tuned_model(checkpoint_path: str):
    model = new_model(ModelConfig())

    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint_dict["model_state_dict"], strict=False)

    loaded_model_cfg = checkpoint_dict["hyperparameters"]["model_cfg"]

    assert loaded_model_cfg.num_classes == model.config.num_classes
    assert (
        loaded_model_cfg.mask_hidden_layer_size == model.config.mask_hidden_layer_size
    )

    set_height_prediction_head(
        model,
        loaded_model_cfg.building_height_pred_class,
        loaded_model_cfg.building_height_pred_loss_fn,
        loaded_model_cfg.height_pred_after_roi_pool,
        loaded_model_cfg.sample_equal,
    )

    # TODO: rm
    model.config.building_height_pred_class = EnhancedTwoMLPRegression

    return model


def show_prediction_results(
    model,
    inf_images,
    inf_targets,
    mask_threshold=0.7,
    box_score_threshold=0.7,
    font=None,
    font_size=18,
):
    model.eval()
    predictions = model(inf_images)

    transform = T.Compose(
        [
            T.Lambda(lambda x: x * 255.0),  # Scale to [0, 255]
            T.Lambda(lambda x: x.to(torch.uint8)),  # Convert to uint8
        ]
    )

    for i, pred in enumerate(predictions):
        img = transform(inf_images[i])

        pred_scores = pred["scores"]
        selected_ids = torch.where(pred_scores > box_score_threshold)[0]
        pred_boxes = pred["boxes"][selected_ids].long()
        pred_scores = pred_scores[selected_ids]
        pred_masks = pred["masks"][selected_ids].squeeze(1)
        # pred_masks = (pred["masks"] > mask_threshold).squeeze(1)
        pred_masks = pred_masks > mask_threshold

        if pred.get("heights") is not None:
            pred_labels = pred["heights"][selected_ids]
            pred_labels = [
                f"height: {height:.1f} ({score:.3f})"
                for height, score in zip(pred_labels, pred_scores)
            ]
        else:
            pred_labels = pred["labels"][selected_ids]
            pred_labels = [
                f"building: {label:.1f} ({score:.3f})"
                for label, score in zip(pred_labels, pred_scores)
            ]

        target_masks = inf_targets[i]["masks"]
        target_boxes = inf_targets[i]["boxes"]
        if pred.get("heights") is not None:
            target_labels = [
                f"height: {h.item()}" for h in inf_targets[i]["building_heights"]
            ]
        else:
            target_labels = [f"building: {h.item()}" for h in inf_targets[i]["labels"]]

        show_segmentation_diffs(
            img,
            pred_masks,
            target_masks,
            pred_boxes,
            target_boxes,
            pred_labels=pred_labels,
            target_labels=target_labels,
            pred_title=f"Prediction with mask threshold {mask_threshold} and score threshold {box_score_threshold}",
            pred_colors="white",
            target_colors="white",
            font=font,
            font_size=font_size,
            # font="/Library/Fonts/Arial Unicode.ttf",
            # font_size=18,
        )

        return predictions


class Checkpoint:
    def __init__(self, root_dir: str | Path, model_name: str) -> None:
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        self.root_dir = root_dir
        self.model_name = model_name

    def load_latest(self) -> Tuple[dict, str]:
        checkpoint_path = self._latest_epoch_checkpoint()

        return torch.load(checkpoint_path), checkpoint_path

    def save(
        self,
        model: nn.Module,
        hyperparameters: dict,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss_dict: dict,
        loss_reduced: float,
    ):
        torch.save(
            {
                # State dicts
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                # Other parameters
                "hyperparameters": hyperparameters,
                "epoch": epoch,
                "loss_reduced": loss_reduced,
                "loss": loss_dict,
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
    model: nn.Module,
    data_loader: DataLoader,
    data_loader_test: DataLoader,
    num_epochs: int,
    checkpoint_dir: str | Path,
    checkpoint_prune_threshold: int,
    eval_iterations: int = 0,
    # Constructors
    new_optimizer: Callable = None,
    new_lr_scheduler: Callable = None,
    print_freq: int = 10,
) -> Tuple[List[float], List[Dict[str, torch.Tensor]]]:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
    checkpoint = Checkpoint(checkpoint_dir, model.config.name)
    try:
        checkpoint_dict, checkpoint_path = checkpoint.load_latest()
        model.load_state_dict(checkpoint_dict["model_state_dict"])
        optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint_dict["lr_scheduler_state_dict"])
        start_epoch = checkpoint_dict["epoch"] + 1
        loss = checkpoint_dict["loss_reduced"]
        logging.info(
            f"Loaded the {checkpoint_path} checkpoint, starting training at the {start_epoch} epoch with reduced loss {loss}",
        )
    except FileNotFoundError:
        logging.info("No checkpoint found, starting from scratch")

    epoch_losses = []
    epoch_loss_dicts = []
    for epoch in range(start_epoch, num_epochs):
        # Train for one epoch, printing every print_freq
        _, losses, loss_dicts = train_one_epoch(
            model,
            optimizer,
            data_loader,
            device,
            epoch,
            num_epochs,
            print_freq=print_freq,
        )

        epoch_losses.append(losses)
        epoch_loss_dicts.append(loss_dicts)

        # Update the learning rate
        lr_scheduler.step()

        # Evaluate on the test dataset
        if eval_iterations != 0:
            mean_iou = evaluate(model, data_loader_test, "cpu", eval_iterations)
            logging.info(f"Mean IoU: {mean_iou}")
        else:
            logging.info("No evaluation")

        logging.info(
            f"Saving model parameters on the {epoch} epoch with loss {losses[-1]}"
        )

        parameters = {
            "model_cfg": model.config,
        }
        checkpoint.save(
            model,
            parameters,
            epoch,
            optimizer,
            lr_scheduler,
            loss_dicts[-1],
            losses[-1],
        )
        checkpoint.prune_old(checkpoint_prune_threshold)

    return epoch_losses, epoch_loss_dicts


# TODO: impl proper evaluation
# def evaluate(model, data_loader, device):
#     n_threads = torch.get_num_threads()
#     # FIXME remove this and make paste_masks_in_image run on the GPU
#     torch.set_num_threads(1)
#     cpu_device = torch.device("cpu")
#     model.eval()

#     metric_logger = local_utils.MetricLogger(delimiter="  ")
#     for images, targets in metric_logger.log_every(data_loader, 100, header="Test:"):
#         images = list(img.to(device) for img in images)

#         if torch.cuda.is_available():
#             torch.cuda.synchronize()

#         model_time = time.time()

#         outputs = model(images)
#         outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
#         model_time = time.time() - model_time

#         evaluator_time = time.time()
#         res = {target["image_id"]: output for target, output in zip(targets, outputs)}
#         logging.debug(f"Results: {res}")
#         # TODO: Run evaluator
#         evaluator_time = time.time() - evaluator_time

#         metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

#     torch.set_num_threads(n_threads)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    eval_iterations: int,
) -> float:
    """
    Evaluate a segmentation model.

    Returns:
        float: The mean IoU score for the dataset.
    """
    model.eval()
    model.to(device)

    total_iou = 0
    num_samples = 0

    data_lodaer_iter = iter(data_loader)
    for _ in range(eval_iterations):
        images, targets = next(data_lodaer_iter)

        images = [image.to(device) for image in images]
        targets = [
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]

        outputs = model(images)
        for output, target in zip(outputs, targets):
            pred_masks = output["masks"].squeeze(1).cpu().numpy()
            true_masks = target["masks"].cpu().numpy()

            for pred_mask, true_mask in zip(pred_masks, true_masks):
                pred_mask = (pred_mask > 0.5).astype(float)
                true_mask = true_mask.astype(float)

                # Compute IoU
                iou = jaccard_score(
                    true_mask.flatten(), pred_mask.flatten(), average="binary"
                )
                total_iou += iou
                num_samples += 1

    mean_iou = total_iou / num_samples

    return mean_iou


def test_predict(
    model_cfg: nn.Module,
    checkpoint_path: str,
    data_loader: DataLoader = None,
    img_path=None,
):
    # print(
    #     test_predict(
    #         model_cfg,
    #         "checkpoints/default_model_epoch_1.pt",
    #         data_loader=data_loader,
    #         # "datasets/mlc_training_data/images_annotated/uqpgutrlld.png",
    #     )
    # )

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


def train_pretrained_model(model_name: str):
    model = new_pretrained_model(
        new_model_name=model_name,
        pretrained_checkpoint_path="checkpoints/pretrained_seg_miyazaki_epoch_9.pt",
        height_pred_class=EnhancedTwoMLPRegression,
        height_pred_loss_fn=SmoothL1Loss(),
        height_pred_after_roi_pool=False,
        sample_equal=False,
        map_location="cpu",
    )

    data_loader, data_loader_test = data_loaders(
        "datasets/images_annotated",
        dataset_cls=BuildingDataset,
        train_batch_size=5,
        test_batch_size=1,
        test_split=0.95,
        num_workers=4,
    )

    return train(
        model,
        data_loader,
        data_loader_test,
        num_epochs=3,
        eval_iterations=0,
        checkpoint_dir="checkpoints",
        checkpoint_prune_threshold=3,
    )


if __name__ == "__main__":
    # og_data_loader, og_data_loader_test = data_loaders(
    #     "datasets/images_annotated/",
    #     dataset_cls=BuildingDataset,
    #     get_transform=get_transform,
    #     train_batch_size=13,
    #     test_batch_size=3,
    #     test_split=0.01,
    #     num_workers=8,
    # )
    # og_data_loader_test_inf_iter = iter(og_data_loader_test)
    # next(og_data_loader_test_inf_iter)

    train_pretrained_model("pretrained_model_v1")

    # print(
    #     test_predict(
    #         model_cfg,
    #         "checkpoints/default_model_epoch_1.pt",
    #         data_loader=data_loader_test,
    #         # "datasets/mlc_training_data/images_annotated/uqpgutrlld.png",
    #     )
    # )
