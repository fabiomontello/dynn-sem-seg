import torch

from datasets.coco import get_coco_dataloaders
from engine import val_model
from models.detr import build, load_detr_weights
from util.config import load_config_from_path_arg

if __name__ == "__main__":

    cfg = load_config_from_path_arg(print_config=True)

    train_dataloader, val_dataloader = get_coco_dataloaders(batch_size=8, only_val=True)
    print(f"Dataset size: {len(val_dataloader)} images")

    # # Inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, postprocessors = build(cfg)
    model = load_detr_weights(model)
    model = model.to(device)
    model.eval()

    metrics = val_model(
        model,
        val_dataloader,
        ann_path="./data/coco/annotations/instances_val2017.json",
        device=device,
    )

    # print(metrics)
