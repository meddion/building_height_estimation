import torch.utils
import os
import time
import torch.utils.data
import logging
import re
from pathlib import Path
import utils
from dataset import BuildingDataset, NUMBER_OF_CLASSES
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import v2 as T
from engine import train_one_epoch, evaluate


logging.basicConfig(level=logging.INFO)

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_PRUNE_THRESHOLD = 3  # No more than N checkpoints to keep.
NUMBER_OF_EPOCHS_DEFAULT = 10
TRAIN_BATCH_SIZE = 3
TEST_BATCH_SIZE = 1
TEST_SPLIT = 0.2


# TODO: learn how roi_heads are linked
def get_model_instance_segmentation(num_classes):
    # Load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # Get number of input features for the classifier.
    # This is a size of features that we get from the backbone.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    # And replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    # TODO: find out what scale does
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())

    return T.Compose(transforms)


def sorted_checkpoints() -> list[Path]:
    def extract_number(file_obj: Path):
        match = re.search(r"\d+", file_obj.stem)
        return int(match.group()) if match else 0

    files = list(CHECKPOINT_DIR.glob("model_epoch_*.pt"))

    return sorted(files, key=extract_number)


def prune_old_checkpoints(threshold: int):
    files = sorted_checkpoints()

    if len(files) < threshold:
        return

    for file in files[:-threshold]:
        os.remove(file)


def latest_epoch_checkpoint() -> Path:
    files = sorted_checkpoints()

    if len(files) == 0:
        raise FileNotFoundError

    return files[-1]


def epoch_checkpoint_filepath(epoch) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    return CHECKPOINT_DIR / f"model_epoch_{epoch}.pt"


def train():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    dataset = BuildingDataset(
        "datasets/mlc_training_data/images_annotated/",
        transforms=get_transform(train=True),
    )

    dataset_test = BuildingDataset(
        "datasets/mlc_training_data/images_annotated/",
        transforms=get_transform(train=False),
    )

    # Split the dataset in train and test set.
    # indices = torch.randperm(len(dataset)).tolist()
    # test_split = int(TEST_SPLIT * len(dataset))
    # dataset = torch.utils.data.Subset(dataset, indices[:-test_split])
    # dataset_test = torch.utils.data.Subset(dataset_test, indices[-test_split:])

    # TODO: rm after debuging
    dataset = torch.utils.data.Subset(dataset, [0, 1])
    dataset_test = torch.utils.data.Subset(dataset_test, [2, 3])

    logging.info(
        f"Size of training set {len(dataset)}. Sise of test set: {len(dataset_test)}"
    )

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )

    model = get_model_instance_segmentation(NUMBER_OF_CLASSES)
    model.to(device)

    # Optimizer.
    params = [p for p in model.parameters() if p.requires_grad]
    # The weight_decay parameter in the optimizer adds a penalty to the loss function
    # based on the L2 norm of the weights, encouraging smaller weights.
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # The learning rate scheduler reduces the learning rate by a factor of 0.1 every 3 epochs.
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = NUMBER_OF_EPOCHS_DEFAULT

    start_epoch = 0
    try:
        checkpoint_path = latest_epoch_checkpoint()
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        logging.info(
            f"Loaded the {checkpoint_path} checkpoint, starting training at the {start_epoch} epoch with loss {checkpoint['loss']}",
        )
    except FileNotFoundError:
        logging.info("No checkpoint found, starting from scratch")

    for epoch in range(start_epoch, num_epochs):
        # Train for one epoch, printing every 10 iterations.
        _, latest_loss = train_one_epoch(
            model, optimizer, data_loader, device, epoch, num_epochs, print_freq=10
        )

        # Update the learning rate
        lr_scheduler.step()
        # Evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

        logging.debug(f"Saving model parameters on {epoch} with loss {latest_loss}")
        prune_old_checkpoints(CHECKPOINT_PRUNE_THRESHOLD)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "loss": latest_loss,
            },
            epoch_checkpoint_filepath(epoch),
        )


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


if __name__ == "__main__":
    train()
