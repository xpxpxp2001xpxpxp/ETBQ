from __future__ import annotations

import os
from typing import Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms


CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD = [0.2675, 0.2565, 0.2761]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TinyImageNetValDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_to_idx=None):
        self.val_dir = os.path.join(root_dir, "val")
        self.transform = transform
        self.image_paths = []
        self.labels = []
        if class_to_idx is None:
            wnids_path = os.path.join(root_dir, "wnids.txt")
            with open(wnids_path, "r") as f:
                wnids = [line.strip() for line in f]
            class_to_idx = {wnid: i for i, wnid in enumerate(wnids)}
        val_anno_path = os.path.join(self.val_dir, "val_annotations.txt")
        with open(val_anno_path, "r") as f:
            for line in f:
                img_name, wnid = line.strip().split("\t")[:2]
                if wnid in class_to_idx:
                    self.image_paths.append(os.path.join(self.val_dir, "images", img_name))
                    self.labels.append(class_to_idx[wnid])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def build_classification_loaders(args) -> Tuple[DataLoader, DataLoader, int]:
    dataset = args.dataset.lower()
    workers = min(os.cpu_count() or 1, args.num_workers)

    if dataset == "cifar100":
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])
        train_set = datasets.CIFAR100(args.data_dir, train=True, download=True, transform=train_tf)
        test_set = datasets.CIFAR100(args.data_dir, train=False, download=True, transform=test_tf)
        num_classes = 100

    elif dataset == "tiny-imagenet":
        train_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        test_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        train_set = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=train_tf)
        test_set = TinyImageNetValDataset(args.data_dir, transform=test_tf, class_to_idx=train_set.class_to_idx)
        num_classes = 200

    elif dataset == "imagenet":
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        test_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        train_dir = os.path.join(args.data_dir, "train")
        val_dir = os.path.join(args.data_dir, "val")
        if os.path.isdir(train_dir) and os.path.isdir(val_dir):
            train_set = datasets.ImageFolder(train_dir, transform=train_tf)
            test_set = datasets.ImageFolder(val_dir, transform=test_tf)
        else:
            train_set = datasets.ImageNet(args.data_dir, split="train", transform=train_tf)
            test_set = datasets.ImageNet(args.data_dir, split="val", transform=test_tf)
        num_classes = 1000

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    return train_loader, val_loader, num_classes


def build_calib_loader(train_dataset, batch_size: int, num_samples: int, num_workers: int, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    count = min(num_samples, len(train_dataset))
    indices = torch.randperm(len(train_dataset), generator=generator)[:count].tolist()
    return DataLoader(
        Subset(train_dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(os.cpu_count() or 1, num_workers),
        pin_memory=True,
    )

