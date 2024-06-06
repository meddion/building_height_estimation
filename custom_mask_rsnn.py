from typing import Dict, List, Tuple
import torch.utils
import torch.utils.data
from torch import Tensor
from torchvision.models.detection.mask_rcnn import (
    MaskRCNN,
    resnet50,
    misc_nn_ops,
    overwrite_eps,
    _resnet_fpn_extractor,
    MaskRCNN_ResNet50_FPN_Weights,
)


import warnings
from collections import OrderedDict
from typing import Any


class CustomMaskRCNN(MaskRCNN):
    def forward(
        self, images: List[Tensor], targets: List[Dict[str, Tensor]] | None = None
    ) -> Tuple[Dict[str, Tensor] | List[Dict[str, Tensor]]]:
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(
                            False,
                            f"Expected target boxes to be of type Tensor, got {type(boxes)}.",
                        )

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(
            features, proposals, images.image_sizes, targets
        )

        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn(
                    "RCNN always returns a (Losses, Detections) tuple in scripting"
                )
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)


def custom_maskrcnn_resnet50_fpn(
    **kwargs: Any,
) -> MaskRCNN:
    """
    A simplified copy of torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    with MaskRCNN replced by CustomMaskRCNN.
    """
    weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
    weights_backbone = None
    num_classes = len(weights.meta["categories"])
    progress = True

    trainable_backbone_layers = 3  # Max: 5
    norm_layer = misc_nn_ops.FrozenBatchNorm2d

    backbone = resnet50(
        weights=weights_backbone, progress=progress, norm_layer=norm_layer
    )
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
    model = CustomMaskRCNN(backbone, num_classes=num_classes, **kwargs)

    if weights is not None:
        model.load_state_dict(
            weights.get_state_dict(progress=progress, check_hash=True)
        )
        if weights == MaskRCNN_ResNet50_FPN_Weights.COCO_V1:
            overwrite_eps(model, 0.0)

    return model
