import torch
import torchvision.models as models
from torchvision import transforms
from transformers import BlipProcessor, BlipForConditionalGeneration
import faiss
import numpy as np
import pickle
import io
import os
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import math

INDEX_FILE = "faiss_index.bin"
FILENAMES_FILE = INDEX_FILE.replace('.bin', '.pkl')
DATASET_PATH = "images" 

INDEX = None
MODEL_RESNET = None
PROCESSOR_BLIP = None
MODEL_BLIP = None
FILENAMES = []

app = FastAPI(
    title="Image Authenticity Search API",
    description="High-speed image search using PyTorch, ResNet50, and FAISS."
)

class SearchQuery(BaseModel):
    k: int = 5

def load_index_and_models():
    global INDEX, MODEL_RESNET, PROCESSOR_BLIP, MODEL_BLIP, FILENAMES
    
    try:
        resnet_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET)
        MODEL_RESNET = torch.nn.Sequential(*(list(resnet_model.children())[:-1]))
        MODEL_RESNET.eval() 
    except Exception:
        pass

    try:
        index_path = os.path.join(os.getcwd(), INDEX_FILE)
        filenames_path = os.path.join(os.getcwd(), FILENAMES_FILE)
        
        INDEX = faiss.read_index(index_path)
        with open(filenames_path, 'rb') as f:
            FILENAMES = pickle.load(f)
    except Exception:
        pass
    
    try:
        PROCESSOR_BLIP = BlipProcessor.from_pretrained("Salesforce/blip-itm-base")
        MODEL_BLIP = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-itm-base")
        MODEL_BLIP.eval()
    except Exception:
        pass


def extract_features(img: Image.Image):
    if MODEL_RESNET is None:
        raise RuntimeError("Model not loaded.")
        
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img_tensor = preprocess(img.convert('RGB'))
    img_tensor = img_tensor.unsqueeze(0)
    
    with torch.no_grad():
        features = MODEL_RESNET(img_tensor).squeeze().cpu().numpy()  
    return features


def search_index(query_features: np.ndarray, k: int, dist_max: float = 35000.0):
    if INDEX is None or not FILENAMES:
        raise RuntimeError("Index or filenames not loaded.")
        
    query_vector = np.array([query_features]).astype('float32')
    distances, indices = INDEX.search(query_vector, k)
    
    results = []
    
    if distances.size > 0:
        dist_min = distances[0][0]
        
        for i, dist in zip(indices[0], distances[0]):
            clamped_dist = max(0.0, min(dist, dist_max))  
            
            denominator = math.log1p(dist_max - dist_min)
            if denominator == 0:
                 normalized_dist = 0.0 if dist == dist_min else 1.0
            else:
                 normalized_dist = math.log1p(clamped_dist - dist_min) / denominator

            similarity_score = 100.0 * (1.0 - normalized_dist)
            similarity_score = max(0.0, min(similarity_score, 100.0))
            
            results.append({
                "filename": FILENAMES[i],
                "similarity_percent": float(similarity_score),
                "distance": float(dist)
            })
            
    return results


@app.on_event("startup")
async def startup_event():
    load_index_and_models()
    

@app.get("/")
async def root():
    return {"message": "Image Search API operational."}

@app.post("/search/")
async def search_image_endpoint(
    file: UploadFile = File(...), 
    k: int = 5
):
    
    if INDEX is None or MODEL_RESNET is None:
        raise HTTPException(
            status_code=503, 
            detail="Service not ready."
        )

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        
        query_features = extract_features(img)
        
        results = search_index(query_features, k)
        
        return {"query_filename": file.filename, "results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Processing error.")