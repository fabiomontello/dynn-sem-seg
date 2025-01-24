"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/facebookresearch/detr/blob/main/datasets/coco.py
"""

import os

import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as T
from pycocotools import mask as coco_mask
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pad

from util.misc import nested_tensor_from_tensor_list


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, root, ann_file, transforms, return_masks=False):
        super(CocoDetection, self).__init__(root, ann_file)
        self.cus_transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks=False)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target)
        # img = np.array(img)

        if self.cus_transforms is not None:
            img = self.cus_transforms(img)
            target["size"] = torch.tensor([img.shape[1], img.shape[2]])

        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor(
            [obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno]
        )
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    if image_set == "train":
        return T.Compose(
            [
                T.Resize(800),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    if image_set == "val":
        return T.Compose(
            [
                T.Resize(800),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    raise ValueError(f"unknown {image_set}")


def download_coco_dataset(root_dir="./data/coco", only_val=False):
    """
    Downloads and prepares COCO train and validation datasets automatically.

    Args:
        root_dir: Directory to store the dataset
    """
    os.makedirs(root_dir, exist_ok=True)

    datasets = {
        "train2017": {
            "images_url": "http://images.cocodataset.org/zips/train2017.zip",
            "annotations_url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        },
        "val2017": {
            "images_url": "http://images.cocodataset.org/zips/val2017.zip",
            "annotations_url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        },
    }

    if only_val:
        datasets.pop("train2017")

    for split, urls in datasets.items():
        split_dir = os.path.join(root_dir, split)
        ann_dir = os.path.join(root_dir, "annotations")

        # Download images if not already downloaded
        if not os.path.exists(split_dir):
            print(f"Downloading COCO {split} dataset...")
            torchvision.datasets.utils.download_and_extract_archive(
                url=urls["images_url"],
                download_root=root_dir,
                extract_root=root_dir,
            )

        # Download annotations if not already downloaded
        if not os.path.exists(os.path.join(ann_dir, "instances_train2017.json")):
            print(f"Downloading COCO annotations for {split}...")
            torchvision.datasets.utils.download_url(
                url=urls["annotations_url"],
                root=root_dir,
                filename="annotations.zip",
            )
            torchvision.datasets.utils.extract_archive(
                os.path.join(root_dir, "annotations.zip"), root_dir
            )


def get_coco_dataloaders(root_dir="./data/coco", batch_size=2, only_val=False):
    """
    Prepares train and validation DataLoaders for COCO dataset.

    Args:
        root_dir: Root directory where COCO data is stored.
        batch_size: Batch size for DataLoaders.

    Returns:
        train_loader, val_loader: PyTorch DataLoader instances.
    """
    # Download the dataset if needed
    download_coco_dataset(root_dir, only_val)
    if not only_val:
        train_dataset = CocoDetection(
            root=os.path.join(root_dir, "train2017"),
            ann_file=os.path.join(root_dir, "annotations/instances_train2017.json"),
            transforms=make_coco_transforms("train"),
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
    else:
        train_loader = None

    val_dataset = CocoDetection(
        root=os.path.join(root_dir, "val2017"),
        ann_file=os.path.join(root_dir, "annotations/instances_val2017.json"),
        transforms=make_coco_transforms("val"),
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_loader, val_loader
