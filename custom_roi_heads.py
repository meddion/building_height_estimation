from typing import Dict, List, Tuple
import torch.utils
import torch.utils.data
from torchvision.models.detection.roi_heads import (
    RoIHeads,
    fastrcnn_loss,
    maskrcnn_inference,
    maskrcnn_loss,
    keypointrcnn_inference,
    keypointrcnn_loss,
)
from torch import Tensor
from torchvision.models.detection.mask_rcnn import (
    misc_nn_ops,
)
import logging

logging.basicConfig(level=logging.DEBUG)


class CustomRoIHeads(RoIHeads):
    """
    Add another roi head for regression
    Also suggested something similar here:
    https://github.com/pytorch/vision/issues/2229
    """

    def __init__(
        self,
        height_predictor,
        loss_fn,
        sample_equal,
        # RoiHeads inputs:
        box_roi_pool,
        box_head,
        box_predictor,
        # Faster R-CNN training
        fg_iou_thresh,
        bg_iou_thresh,
        batch_size_per_image,
        positive_fraction,
        bbox_reg_weights,
        # Faster R-CNN inference
        score_thresh,
        nms_thresh,
        detections_per_img,
        # Mask
        mask_roi_pool=None,
        mask_head=None,
        mask_predictor=None,
        keypoint_roi_pool=None,
        keypoint_head=None,
        keypoint_predictor=None,
    ):
        super().__init__(
            box_roi_pool,
            box_head,
            box_predictor,
            fg_iou_thresh,
            bg_iou_thresh,
            batch_size_per_image,
            positive_fraction,
            bbox_reg_weights,
            score_thresh,
            nms_thresh,
            detections_per_img,
            mask_roi_pool,
            mask_head,
            mask_predictor,
            keypoint_roi_pool,
            keypoint_head,
            keypoint_predictor,
        )

        self.height_predictor = height_predictor
        self.sample_equal = sample_equal
        self.loss_fn = loss_fn

    # Ensure equal number of building and not building samples
    def sample_buildings(building_indices, not_building_indices):
        # Ensure equal number of building and not building samples
        num_buildings = len(building_indices)
        num_not_buildings = len(not_building_indices)
        if num_buildings > 0 and num_not_buildings > 0:
            num_samples = min(num_buildings, num_not_buildings)

            # Randomly sample indices
            sampled_building_indices = building_indices[
                torch.randperm(num_buildings)[:num_samples]
            ]
            sampled_not_building_indices = not_building_indices[
                torch.randperm(num_not_buildings)[:num_samples]
            ]

            return torch.cat([sampled_building_indices, sampled_not_building_indices])

        return torch.cat([building_indices, not_building_indices])

    def height_regression_loss(
        self,
        height_predictions: Tensor,
        matched_idxs: List[Tensor],
        labels: List[Tensor],
        gt_heights: List[Tensor],
    ) -> Tensor:
        # TODO: simplify
        proposal_heights = []
        # Combine ground truth heights based on matched indices
        proposal_heights = [
            gt_heights[i][matched_idx] for i, matched_idx in enumerate(matched_idxs)
        ]
        proposal_heights = torch.cat(proposal_heights).to(height_predictions.device)

        # Combine labels into a single tensor
        labels = torch.cat(labels).to(height_predictions.device)

        # Get indices of buildings (label == 1) and not buildings (label == 0)
        building_indices = torch.where(labels == 1)[0]
        not_building_indices = torch.where(labels == 0)[0]

        # Set height to 0 for not buildings
        proposal_heights[not_building_indices] = 0
        proposal_heights = proposal_heights.to(height_predictions.dtype)

        if self.sample_equal:
            sampled_indices = self.sample_buildings(
                building_indices, not_building_indices
            )
            height_predictions = height_predictions[sampled_indices]
            proposal_heights = proposal_heights[sampled_indices]

        return self.loss_fn(height_predictions, proposal_heights)

    def forward(
        self,
        features: Dict[str, Tensor],
        proposals: List[misc_nn_ops.Tensor],
        image_shapes: List[Tuple[int]],
        targets: List[Dict[str, Tensor]] | None = None,
    ) -> Tuple[List[Dict[str, Tensor]] | Dict[str, Tensor]]:
        """Copy paste of RoIHeads.forward with an additional regression head."""
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                if t["boxes"].dtype not in floating_point_types:
                    raise TypeError(
                        f"target boxes must of float type, instead got {t['boxes'].dtype}"
                    )
                if not t["labels"].dtype == torch.int64:
                    raise TypeError(
                        f"target labels must of int64 type, instead got {t['labels'].dtype}"
                    )
                if self.has_keypoint():
                    if not t["keypoints"].dtype == torch.float32:
                        raise TypeError(
                            f"target keypoints must of float type, instead got {t['keypoints'].dtype}"
                        )

        if self.training:
            proposals, matched_idxs, labels, regression_targets = (
                self.select_training_samples(proposals, targets)
            )
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        height_predictions = self.height_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            if labels is None:
                raise ValueError("labels cannot be None")
            if regression_targets is None:
                raise ValueError("regression_targets cannot be None")
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets
            )

            building_height_loss = self.height_regression_loss(
                height_predictions,
                matched_idxs,
                labels,
                [t["building_heights"] for t in targets],
            )

            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg,
                "loss_building_height_reg": building_height_loss,
            }
        else:
            # TODO: add height prediction to results
            boxes, scores, labels = self.postprocess_detections(
                class_logits, box_regression, proposals, image_shapes
            )
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                if matched_idxs is None:
                    raise ValueError("if in training, matched_idxs should not be None")

                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(
                    features, mask_proposals, image_shapes
                )
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
            else:
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            if self.training:
                if targets is None or pos_matched_idxs is None or mask_logits is None:
                    raise ValueError(
                        "targets, pos_matched_idxs, mask_logits cannot be None when training"
                    )

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                rcnn_loss_mask = maskrcnn_loss(
                    mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs
                )
                loss_mask = {"loss_mask": rcnn_loss_mask}
            else:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)

        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        if (
            self.keypoint_roi_pool is not None
            and self.keypoint_head is not None
            and self.keypoint_predictor is not None
        ):
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                if matched_idxs is None:
                    raise ValueError("if in trainning, matched_idxs should not be None")

                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            keypoint_features = self.keypoint_roi_pool(
                features, keypoint_proposals, image_shapes
            )
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                if targets is None or pos_matched_idxs is None:
                    raise ValueError(
                        "both targets and pos_matched_idxs should not be None when in training mode"
                    )

                gt_keypoints = [t["keypoints"] for t in targets]
                rcnn_loss_keypoint = keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals, gt_keypoints, pos_matched_idxs
                )
                loss_keypoint = {"loss_keypoint": rcnn_loss_keypoint}
            else:
                if keypoint_logits is None or keypoint_proposals is None:
                    raise ValueError(
                        "both keypoint_logits and keypoint_proposals should not be None when not in training mode"
                    )

                keypoints_probs, kp_scores = keypointrcnn_inference(
                    keypoint_logits, keypoint_proposals
                )
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps
            losses.update(loss_keypoint)

        return result, losses
