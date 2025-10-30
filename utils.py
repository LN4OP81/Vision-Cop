# utils.py

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import imagehash
import math
import os

# Assuming DATASET_PATH is imported from config
from config import DATASET_PATH 

# --- UTILITY FUNCTIONS ---

def get_image_filenames():
    """Gets the sorted list of filenames from the dataset folder."""
    if os.path.exists(DATASET_PATH):
        # NOTE: Updated to include common image formats like the previous version
        return sorted([f for f in os.listdir(DATASET_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    return []

def extract_features(img, model_resnet):
    """Extracts a feature vector using ResNet50."""
    preprocess = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(img.convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        features = model_resnet(img_tensor).squeeze().cpu().numpy() 
    return features

def search_index(index, query_features, image_ids, k):
    """Searches the FAISS index and calculates similarity scores."""
    query_vector = np.array([query_features]).astype('float32')
    distances, indices = index.search(query_vector, k)
    results = []
    
    if distances.size > 0:
        dist_min = distances[0][0]
        dist_max = 35000.0
        
        for i, dist in zip(indices[0], distances[0]):
            # Similarity scaling logic from the original code
            clamped_dist = max(0.0, min(dist, dist_max))
            normalized_dist = math.log1p(clamped_dist - dist_min) / math.log1p(dist_max - dist_min)
            similarity_score = 100.0 * (1.0 - normalized_dist)
            similarity_score = max(0.0, min(similarity_score, 100.0))
            
            results.append({
                "id": i, "filename": image_ids[i], "similarity_percent": similarity_score, "distance": dist
            })
    return results

def verify_phash(query_img, original_path):
    """Performs Perceptual Hashing check for manipulation/re-use."""
    try:
        query_hash = imagehash.phash(query_img)
        original_img = Image.open(original_path)
        original_hash = imagehash.phash(original_img)
        
        # Check against a mirrored version to detect simple flips
        flipped_original_img = original_img.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_original_hash = imagehash.phash(flipped_original_img)
        
        distance_original = query_hash - original_hash
        distance_flipped = query_hash - flipped_original_hash
        
        return distance_original, distance_flipped
    except Exception:
        # Return error signal as per original code's logic
        return -1, -1

def generate_tags(image_path, processor_blip, model_blip):
    """Generates a caption and keywords for an image using the BLIP VLM."""
    try:
        img = Image.open(image_path).convert('RGB')
        inputs = processor_blip(img, return_tensors="pt")
        out = model_blip.generate(**inputs, max_length=30)
        caption = processor_blip.decode(out[0], skip_special_tokens=True)
        tags = [word.strip(',.') for word in caption.lower().split() if len(word) > 3]
        return tags, caption
    except Exception:
        return ["error", "generation_failed"], "VLM could not generate a caption for this image."

def check_metadata_mismatch(query_context, original_metadata):
    """Flags inconsistencies between query claim and VLM-generated original data."""
    flags = []
    query_loc = query_context.get('location', '').lower()
    
    # Location Mismatch Check
    if query_loc and "auto-generated" in original_metadata.get('location', '').lower():
         flags.append(f"LOCATION MISMATCH: Claimed '{query_context['location']}' is highly specific, but source data is AI-generated and general. Verify source independently.")
        
    # Chronological Mismatch Check
    query_year = query_context.get('year')
    original_year = original_metadata.get('year')
    if query_year and original_year and abs(query_year - original_year) >= 3:
        flags.append(f"CHRONOLOGICAL MISMATCH: Claimed year {query_year} is too far from VLM's baseline year {original_year}.")

    # Tag Mismatch Check (Disaster vs Leisure)
    if 'tags' in original_metadata:
        query_keywords = set(query_context.get('tag', '').lower().split())
        disaster_words = set(['fire', 'disaster', 'riot', 'bomb', 'attack', 'explosion', 'conflict', 'emergency'])
        leisure_words = set(['holiday', 'leisure', 'beach', 'pet', 'fun', 'park', 'sunset', 'water'])
        
        if any(w in query_keywords for w in disaster_words) and any(t in original_metadata['tags'] for t in leisure_words):
             flags.append("TAG MISMATCH: Query suggests conflict/disaster, but VLM tags suggest a leisure/calm scene.")
             
    return flags