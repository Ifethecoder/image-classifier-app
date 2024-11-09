import torch
from torchvision import models
from torch import nn

def build_model(arch="vgg16", hidden_units=256):
    """Builds a model based on the specified architecture and hidden units."""
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
    elif arch == "vgg13":
        model = models.vgg13(pretrained=True)
    else:
        raise ValueError("Unsupported architecture")

    # Freeze feature parameters
    for param in model.parameters():
        param.requires_grad = False

    # Define the new classifier
    input_units = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(input_units, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )

    return model

def load_checkpoint(filepath):
    """Loads a model from a checkpoint file."""
    checkpoint = torch.load(filepath)
    model = build_model(checkpoint['arch'], checkpoint['hidden_units'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model
