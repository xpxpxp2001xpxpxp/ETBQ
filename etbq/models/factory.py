from __future__ import annotations

import torch.nn as nn
import torchvision.models as tv_models

try:
    import timm
except ImportError:  # pragma: no cover
    timm = None


def _patch_resnet_for_cifar(model: nn.Module) -> nn.Module:
    model.conv1 = nn.Conv2d(3, model.conv1.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def _replace_classifier(model: nn.Module, arch: str, num_classes: int) -> nn.Module:
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            last = model.classifier[-1]
            if isinstance(last, nn.Linear):
                model.classifier[-1] = nn.Linear(last.in_features, num_classes)
        elif isinstance(model.classifier, nn.Linear):
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def _torchvision_model(arch: str, num_classes: int, pretrained: bool) -> nn.Module:
    weight_map = {
        "resnet18": (tv_models.resnet18, "ResNet18_Weights"),
        "resnet50": (tv_models.resnet50, "ResNet50_Weights"),
        "mobilenet_v2": (tv_models.mobilenet_v2, "MobileNet_V2_Weights"),
    }
    fn, weights_name = weight_map[arch]
    try:
        weights_cls = getattr(tv_models, weights_name)
        model = fn(weights=weights_cls.DEFAULT if pretrained else None)
    except (AttributeError, TypeError):
        model = fn(pretrained=pretrained)
    return _replace_classifier(model, arch, num_classes)


def build_classification_model(
    arch: str,
    num_classes: int,
    dataset: str,
    pretrained: bool = False,
) -> nn.Module:
    arch = arch.lower()
    dataset = dataset.lower()

    if arch in {"resnet18", "resnet50", "mobilenet_v2"}:
        model = _torchvision_model(arch, num_classes, pretrained=pretrained)
        if dataset == "cifar100" and arch.startswith("resnet"):
            model = _patch_resnet_for_cifar(model)
        return model

    if arch in {"mobilenet_v1", "ghostnet_100", "ghostnet_130", "ghostnetv2_100"}:
        if timm is None:
            raise ImportError(f"{arch} requires timm. Install with: pip install timm")
        timm_name = "mobilenetv1_100" if arch == "mobilenet_v1" else arch
        return timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)

    raise ValueError(f"Unsupported architecture: {arch}")

