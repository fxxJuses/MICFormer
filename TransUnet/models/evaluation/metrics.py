def iou_score(preds, targets, eps=1e-7):
    """
    Compute the Intersection over Union (IoU)

    Args:
        preds (torch.Tensor): 
            Predictions probabilities from the model (B, C, D, H, W)
        targets (torch.Tensor):
            Ground truth labels (B, C, D, H, W)
        eps (float):
            Small value to avoid division by zero
    """
    # Flatten the predictions and targets
    preds = preds.view(-1)
    targets = targets.view(-1)

    # Compute the intersection and union
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection

    # Compute the IoU score and return the average across all classes and batches
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()