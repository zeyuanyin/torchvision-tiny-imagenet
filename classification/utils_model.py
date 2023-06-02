import torch
import torch.nn as nn
import torchvision

def load_ckp(model, path):
    checkpoint = torch.load(path, map_location="cpu")
    if "model" in checkpoint:
        checkpoint = checkpoint["model"]
    elif "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if "module." in list(checkpoint.keys())[0]:
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)
    return model


def load_model(model_name, num_classes, weights=None, path=None, device="cuda"):
    if num_classes == 1000:
        model = torchvision.models.get_model(model_name, weights=weights)
    else: # tinyimagenet
        model = torchvision.models.get_model(model_name, num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()

    if path is not None: # checkpoint will replace weights if both are provided
        model = load_ckp(model, path)

    model.to(device)
    return model
