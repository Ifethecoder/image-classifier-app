import argparse
import torch
from model import load_checkpoint 
from utils import process_image  
import json
import numpy as np

def predict(image_path, checkpoint, top_k, category_names, gpu):
    # Load checkpoint and rebuild model
    model = load_checkpoint(checkpoint)  
    class_to_idx = model.class_to_idx 
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Process the image
    image = process_image(image_path).unsqueeze(0).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Forward pass to get predictions
    with torch.no_grad():
        output = model(image)
    probs, indices = torch.exp(output).topk(top_k) 
    
    # Map indices to classes
    idx_to_class = {v: k for k, v in class_to_idx.items()}  # Reverse class_to_idx to get idx_to_class
    classes = [idx_to_class[idx] for idx in indices.cpu().numpy().squeeze()]  # Convert indices to class labels
    
    # Convert class labels to names if category_names is provided
    if category_names:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        class_names = [cat_to_name[str(cls)] for cls in classes]
    else:
        class_names = classes

    return probs.cpu().numpy().squeeze(), class_names

# Argument parser setup
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the class of an image.")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("checkpoint", type=str, help="Model checkpoint file")
    parser.add_argument("--top_k", type=int, default=5, help="Return top K predictions")
    parser.add_argument("--category_names", type=str, help="Path to category names JSON file")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")

    args = parser.parse_args()
    
    # Call predict with the parsed arguments
    probs, classes = predict(args.image_path, args.checkpoint, args.top_k, args.category_names, args.gpu)
    
    # Print the results
    for prob, cls in zip(probs, classes):
        print(f"{cls}: {prob:.4f}")
