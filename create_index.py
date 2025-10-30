# create_index.py - Builds the FAISS searchable index

import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import faiss
import os

# --- CONFIGURATION (Must match the app.py settings) ---
DATASET_PATH = 'images'  # Folder containing your MIRFlickr images (or subset)
INDEX_FILE = 'faiss_index.bin'
VECTOR_DIM = 2048 # ResNet50 output dimension

# --- Feature Extraction Setup ---
# Load model once at the start of the script
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.eval() 

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_path):
    """Extracts a 2048-dimensional feature vector using ResNet50."""
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = preprocess(img)
        img_tensor = img_tensor.unsqueeze(0)
        
        with torch.no_grad():
            features = model(img_tensor).squeeze().cpu().numpy()
        return features
    except Exception as e:
        # print(f"Error processing {image_path}: {e}") # Suppress this print during indexing for clean output
        return None

def create_faiss_index(dataset_path):
    """Creates a FAISS index from all dataset images."""
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset folder '{dataset_path}' not found.")
        return

    image_paths = sorted([os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if not image_paths:
        print(f"Error: No images found in the dataset path: {dataset_path}")
        return

    all_features = []
    print(f"Extracting features from {len(image_paths)} images...")
    
    for i, path in enumerate(image_paths):
        features = extract_features(path)
        if features is not None:
            all_features.append(features)
            
    if not all_features:
        print("No valid features extracted.")
        return

    feature_matrix = np.array(all_features).astype('float32')
    dimension = feature_matrix.shape[1]
    
    # Initialize and populate the FAISS index
    index = faiss.IndexFlatL2(dimension)
    index.add(feature_matrix)
    
    faiss.write_index(index, INDEX_FILE)
    print(f"Success! Index created and saved to {INDEX_FILE}. Total vectors: {index.ntotal}")

# --- Main Execution ---
if __name__ == '__main__':
    print("Starting FAISS Index Creation...")
    create_faiss_index(DATASET_PATH)