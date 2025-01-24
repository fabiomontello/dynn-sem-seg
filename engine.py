import torch
import torch.nn as nn
import torch.optim as optim
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm


def box_cxcywh_to_xyxy(x):
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2)"""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_xywh(x):
    """Convert bbox coordinates from (x1, y1, x2, y2) to (x, y, w, h)"""
    x1, y1, x2, y2 = x.unbind(-1)
    b = [x1, y1, (x2 - x1), (y2 - y1)]
    return torch.stack(b, dim=-1)


# Training loop
def train_model(model, dataloader, num_epochs=10, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}")


def val_model(model, val_loader, ann_path, confidence_threshold=0.5, device="cuda"):
    """
    Evaluate DETR model using COCO evaluation framework

    Args:
        model: DETR model
        val_loader: validation dataloader
        ann_path: path to COCO annotation file
        confidence_threshold: confidence threshold for predictions
        device: device to run evaluation on

    Returns:
        dict with COCO evaluation metrics
    """
    model.eval()
    results = []
    image_ids = []
    coco_gt = COCO(ann_path)

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            # img_size = images[0].shape[-2:]
            img_size = (800, 666)
            # Process model outputs
            processed_outputs = postprocess_outputs(
                outputs, targets, confidence_threshold, img_size
            )

            results.extend(processed_outputs)
            image_ids.extend([t["image_id"].item() for t in targets])

    # Convert results to COCO format
    coco_results = results

    # Perform COCO evaluation
    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Return metrics
    return {
        "AP": coco_eval.stats[0],
        "AP50": coco_eval.stats[1],
        "AP75": coco_eval.stats[2],
        "APs": coco_eval.stats[3],
        "APm": coco_eval.stats[4],
        "APl": coco_eval.stats[5],
    }


def postprocess_outputs(outputs, targets, confidence_threshold, img_size):
    """
    Post-process DETR model outputs

    Args:
        outputs: model predictions
        targets: ground truth targets
        confidence_threshold: minimum confidence for keeping predictions

    Returns:
        List of processed predictions
    """
    processed_results = []

    # DETR outputs typically contain 'pred_logits' and 'pred_boxes'
    pred_logits = outputs["pred_logits"]
    pred_boxes = outputs["pred_boxes"]

    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(pred_logits, dim=-1)

    # Get max probabilities and corresponding class indices
    max_probs, class_indices = probabilities[:, :, :-1].max(dim=-1)

    for i in range(len(targets)):
        # Filter predictions by confidence threshold
        valid_indices = max_probs[i] > confidence_threshold

        # Get valid predictions
        valid_boxes = pred_boxes[i][valid_indices]
        valid_classes = class_indices[i][valid_indices]
        valid_scores = max_probs[i][valid_indices]

        # Convert to original image coordinates if needed
        orig_size = targets[i]["orig_size"]

        # Rescale boxes to original image size
        valid_boxes = box_cxcywh_to_xyxy(valid_boxes)
        valid_boxes = rescale_bboxes(valid_boxes, orig_size)
        valid_boxes = box_xyxy_to_xywh(valid_boxes)

        # Create processed predictions for each valid detection
        for box, cls, score in zip(valid_boxes, valid_classes, valid_scores):
            processed_results.append(
                {
                    "image_id": targets[i]["image_id"].item(),
                    "category_id": cls.item(),
                    "bbox": box.tolist(),
                    "score": score.item(),
                }
            )

    return processed_results


def rescale_bboxes(boxes, size):
    """
    Rescale bounding boxes from model's input size to original image size
    """
    h, w = size

    scale_fct = torch.tensor([w, h, w, h], device=boxes.device)
    boxes = boxes * scale_fct

    return boxes
