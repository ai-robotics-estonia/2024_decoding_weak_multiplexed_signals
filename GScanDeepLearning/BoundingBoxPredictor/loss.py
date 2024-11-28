import torch
import torch.nn as nn
from torchvision.ops import box_iou

# class CustomMaskedMSELoss(nn.Module):
#     def __init__(self):
#         super(CustomMaskedMSELoss, self).__init__()
#         self.mse_loss = nn.MSELoss(reduction='none')  # Use 'none' to calculate loss per element

#     def forward(self, predictions, targets):
#         # Calculate MSE loss for all elements
#         loss = self.mse_loss(predictions, targets)
        
#         # Create a mask to ignore elements where the target is [-1, -1, -1, -1]
#         mask = (targets != -1).all(dim=-1)  # Shape: (batch_size, 6)
        
#         # Apply the mask to the loss (only valid elements)
#         masked_loss = loss[mask.unsqueeze(-1).expand_as(loss)]
        
#         # Calculate the mean loss over valid elements only
#         return masked_loss.mean()
    
class CustomMaskedMSELoss(nn.Module):
    def __init__(self, center_weight=5.0, size_weight=1.0):
        super(CustomMaskedMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')  # Use 'none' to calculate loss per element
        self.center_weight = center_weight
        self.size_weight = size_weight

    def forward(self, predictions, targets):
        # Calculate MSE loss for all elements
        loss = self.mse_loss(predictions, targets)
        
        # Apply different weights to center and size elements
        weights = torch.tensor([self.center_weight, self.center_weight, self.size_weight, self.size_weight], device=predictions.device)
        weighted_loss = loss * weights

        # Create a mask to ignore elements where the target is [-1, -1, -1, -1]
        mask = (targets != -1).all(dim=-1)
        
        # Apply the mask to the weighted loss (only valid elements)
        masked_loss = weighted_loss[mask.unsqueeze(-1).expand_as(loss)]
        
        # Calculate the mean loss over valid elements only
        return masked_loss.mean()

import torch
import torch.nn as nn

# class CustomIOU(nn.Module):
#     def __init__(self):
#         super(CustomIOU, self).__init__()

#     def convert_center_to_corner(self, boxes):
#         """
#         Convert boxes from (center_x, center_y, width, height) to (x1, y1, x2, y2)
#         """
#         center = boxes[..., :2]
#         size = boxes[..., 2:]
#         corner1 = center - size / 2
#         corner2 = center + size / 2
#         return torch.cat([corner1, corner2], dim=-1)

#     def compute_one_to_one_iou(self, boxes1, boxes2):
#         """
#         Compute IoU for corresponding boxes in boxes1 and boxes2 on a 1-to-1 basis.
        
#         Args:
#             boxes1: Tensor of shape [batch_size, 6, 4] in (x1, y1, x2, y2) format
#             boxes2: Tensor of shape [batch_size, 6, 4] in (x1, y1, x2, y2) format
        
#         Returns:
#             iou: Tensor of shape [batch_size, 6] containing IoU values for corresponding boxes
#         """
#         # Compute intersection coordinates
#         inter_x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
#         inter_y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
#         inter_x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
#         inter_y2 = torch.min(boxes1[..., 3], boxes2[..., 3])
        
#         # Compute intersection area
#         inter_w = (inter_x2 - inter_x1).clamp(min=0)
#         inter_h = (inter_y2 - inter_y1).clamp(min=0)
#         inter_area = inter_w * inter_h
        
#         # Compute areas of individual boxes
#         area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
#         area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        
#         # Compute union area
#         union_area = area1 + area2 - inter_area
        
#         # Compute IoU
#         iou = inter_area / union_area.clamp(min=1e-6)  # Avoid division by zero
#         return iou

#     def forward(self, boxes1, boxes2):
#         """
#         Compute 1-to-1 IoU for two sets of boxes with possible empty boxes.
        
#         Args:
#             boxes1: Tensor of shape [batch_size, 6, 4] in (center_x, center_y, width, height) format
#             boxes2: Tensor of shape [batch_size, 6, 4] in (center_x, center_y, width, height) format
        
#         Returns:
#             mean_iou: Average IoU across all valid box pairs in the batch on a 1-to-1 basis
#         """
#         # Create masks to filter out invalid boxes
#         valid_mask1 = ~torch.all(boxes1 == -1, dim=-1)  # [batch_size, 6]
#         valid_mask2 = ~torch.all(boxes2 == -1, dim=-1)  # [batch_size, 6]
#         valid_mask = valid_mask1 & valid_mask2  # Only where both boxes are valid

#         # Convert boxes to corner coordinates
#         boxes1_corners = self.convert_center_to_corner(boxes1)
#         boxes2_corners = self.convert_center_to_corner(boxes2)
        
#         # Compute IoU for valid boxes in a 1-to-1 manner
#         iou_matrix = self.compute_one_to_one_iou(boxes1_corners, boxes2_corners)  # [batch_size, 6]

#         # Apply mask to IoU values to ignore invalid boxes
#         valid_iou_values = iou_matrix[valid_mask]
        
#         # Compute the mean IoU over all valid pairs
#         mean_iou = valid_iou_values.mean() if valid_iou_values.numel() > 0 else torch.tensor(0.0)
        
#         return mean_iou



class CustomIOU(nn.Module):
    def __init__(self):
        super(CustomIOU, self).__init__()



    def intersection_over_union(self,boxes_preds, boxes_labels, box_format="midpoint"):
        """
        Calculates intersection over union
    
        Parameters:
            boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
            boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
            box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    
        Returns:
            tensor: Intersection over union for all examples
        """
    
        # Slicing idx:idx+1 in order to keep tensor dimensionality
        # Doing ... in indexing if there would be additional dimensions
        # Like for Yolo algorithm which would have (N, S, S, 4) in shape
        if box_format == "midpoint":

            box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
            box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
            box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
            box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
            box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
            box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
            box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
            box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
    
        elif box_format == "corners":
            box1_x1 = boxes_preds[..., 0:1]
            box1_y1 = boxes_preds[..., 1:2]
            box1_x2 = boxes_preds[..., 2:3]
            box1_y2 = boxes_preds[..., 3:4]
            box2_x1 = boxes_labels[..., 0:1]
            box2_y1 = boxes_labels[..., 1:2]
            box2_x2 = boxes_labels[..., 2:3]
            box2_y2 = boxes_labels[..., 3:4]
    
        x1 = torch.max(box1_x1, box2_x1)
        y1 = torch.max(box1_y1, box2_y1)
        x2 = torch.min(box1_x2, box2_x2)
        y2 = torch.min(box1_y2, box2_y2)
    
        # Need clamp(0) in case they do not intersect, then we want intersection to be 0
        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
        box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    
        return intersection / (box1_area + box2_area - intersection + 1e-6)


    def forward(self, pred, truth):
        batch_size = pred.size(0)
        total_iou = 0.0
        valid_boxes = 0

        for bat in range(batch_size):
            for box_in_sample in range(6):
                # Check if the box in truth is valid (not [-1, -1, -1, -1])
                if torch.all(truth[bat, box_in_sample] == torch.tensor([-1, -1, -1, -1], device=truth.device)):
                    continue
                else:
                    predict=pred[bat, box_in_sample].unsqueeze(0)
                    label=truth[bat, box_in_sample].unsqueeze(0)

                    # Calculate IoU between corresponding predicted and true box
                    iou = self.intersection_over_union(predict,label)
                    total_iou += iou
                    valid_boxes += 1

        # Return mean IoU across all valid boxes
        mean_iou = total_iou / valid_boxes if valid_boxes > 0 else torch.tensor(0.0, device=truth.device)
        return mean_iou