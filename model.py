import torch.utils
import torch.utils.data
import utils
from dataset import BuildingDataset, NUMBER_OF_CLASSES
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import v2 as T
from engine import train_one_epoch, evaluate


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
    indices = torch.randperm(len(dataset)).tolist()
    test_split = int(0.2 * len(dataset))
    dataset = torch.utils.data.Subset(dataset, indices[:-test_split])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-test_split:])
    print("Size of training set:", len(dataset), "Sise of test set:", len(dataset_test))

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )

    model = get_model_instance_segmentation(NUMBER_OF_CLASSES)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    # The weight_decay parameter in the optimizer adds a penalty to the loss function
    # based on the L2 norm of the weights, encouraging smaller weights.
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # The learning rate scheduler reduces the learning rate by a factor of 0.1 every 3 epochs.
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 2

    for epoch in range(num_epochs):
        # Train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # Update the learning rate
        lr_scheduler.step()
        # Evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)


if __name__ == "__main__":
    train()
