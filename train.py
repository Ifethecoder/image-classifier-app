import argparse
import torch
from model import build_model
from utils import load_data

def train_model(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu):
    # Load the data
    trainloader, validloader, testloader, train_data = load_data(data_dir)
    
    # Build the model
    model = build_model(arch, hidden_units)
    
    # Set device
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Define optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.NLLLoss()

    
    # Training loop
    # Adapted from Deep Learning with PyTorch Course
    steps = 0
    running_loss = 0
    print_every = 15
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(testloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(testloader):.3f}")
                running_loss = 0
                model.train()
    
    model.class_to_idx = train_data.class_to_idx
    # Save checkpoint
    checkpoint = {
        'arch': arch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epochs,
        'class_to_idx': model.class_to_idx,
        'hidden_units': hidden_units
    }
    torch.save(checkpoint, f"{save_dir}/checkpoint.pth")
    print(f"Model saved to {save_dir}/checkpoint.pth")

# Argument parser setup
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network on a dataset.")
    parser.add_argument("data_dir", type=str, help="Directory of the dataset")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--arch", type=str, default="vgg16", help="Model architecture (e.g., 'vgg16')")
    parser.add_argument("--learning_rate", type=float, default=0.003, help="Learning rate for the optimizer")
    parser.add_argument("--hidden_units", type=int, default=256, help="Number of hidden units")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")

    args = parser.parse_args()
    
    # Call train_model with the parsed arguments
    train_model(args.data_dir, args.save_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu)

