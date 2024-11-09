# Image Classifier Application 

This project is a command-line application built with PyTorch and allows users to create their own model, save model checkpoints, and use model to make predictions. It also includes options for customizing model architecture, hyperparameters, and device (CPU/GPU) usage.

## Project Features
- **Train a model** with customizable parameters.
- **Predict the class** of new images with top-K probabilities.
- **Save and load model checkpoints** for flexibility and reuse.

## Getting Started

### 1. Clone the Repository  
Clone the repository and navigate into the project directory:  
git clone https://github.com/Ifethecoder/image-classifier-app.git  
cd image-classifier-app

### 2. Install Dependencies  
Make sure Python 3.x is installed, then install dependencies from requirements.txt:    
pip install -r requirements.txt  

### 3. Prepare the Dataset  
This application uses images structured in the following format, compatible with torchvision.datasets.ImageFolder:    

data/  
├── train/  
├── valid/  
└── test/  

### 4. Running the Application
#### Training the Model
You can train the model on your dataset with custom parameters. Example:  
python train.py data --save_dir checkpoints --arch vgg16 --learning_rate 0.01 --hidden_units 512 --epochs 20 --gpu  

#### Training Arguments:
data: Path to the dataset directory.  
--save_dir: Directory to save the trained model checkpoint.  
--arch: Model architecture (vgg13 or vgg16).  
--learning_rate: Learning rate for the optimizer.  
--hidden_units: Number of hidden units in the classifier layer.  
--epochs: Number of epochs for training.  
--gpu: Use GPU for training if available.  

#### Making Predictions
To predict the class of a new image, specify the image path and a trained checkpoint:  
python predict.py /path/to/image checkpoints/checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu  

#### Prediction Arguments:
/path/to/image: Path to the image to classify.  
checkpoints/checkpoint.pth: Model checkpoint file.  
--top_k: Number of top predictions to return (default: 5).  
--category_names: Path to JSON file mapping class indices to names.  
--gpu: Use GPU for inference if available.  

## Note: 
A sample dataset of flower images is provided with the project, along with a JSON file that maps category labels to flower names. You can use this dataset to train your own model if desired.




