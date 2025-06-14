Deepfake Detection using Vision Transformers (ViT)
This project uses a Vision Transformer (ViT) model for detecting deepfake content from video frames. It includes steps for video frame extraction, data preprocessing, model training, and evaluation. The dataset used in this notebook is based on the DFD (Deepfake Detection) dataset, and the model achieves strong performance by leveraging frame-level predictions.

📁 Dataset
The notebook uses:

Real Videos Directory:
/kaggle/input/deep-fake-detection-dfd-entire-original-dataset/DFD_original sequences

Manipulated Videos Directory:
/kaggle/input/deep-fake-detection-dfd-entire-original-dataset/DFD_manipulated_sequences/DFD_manipulated_sequences

These videos are preprocessed to extract frames, which are then used to train the ViT model.

🔧 Features
Extracts frames from real and manipulated videos.

Applies image transformations for data augmentation.

Uses Vision Transformer (via timm library) for classification.

Implements frame-level voting to improve video-level predictions.

Runs efficiently on GPU (if available).

📦 Dependencies
Install the necessary libraries:

bash
Copy
Edit
pip install timm
Other required libraries include:

torch, torchvision

opencv-python

numpy, pandas

Pillow

🚀 How It Works
Frame Extraction: Frames are extracted from both real and fake videos at 1 frame per second.

Preprocessing: Images are resized and normalized.

Model: A ViT model (e.g., ViT-Large Patch16-224) is loaded using the timm library.

Training: The model is trained using a labeled dataset of frames.

Evaluation: Accuracy is computed on a validation set, with frame-level predictions combined for final output.

🧪 Results
This model has shown strong accuracy in detecting manipulated videos, especially when combined with frame-level voting strategies.

📂 File Structure
php
Copy
Edit
deepfake-detection-vit-model-final.ipynb   # Main notebook
README.md                                  # Project overview and instructions
✅ Future Work
Extend model to support more datasets (e.g., DFDC, Celeb-DF).

Add early stopping and learning rate scheduler.

Improve video-level classification using sequence models (LSTM/Transformer).
