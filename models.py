# models.py

import streamlit as st
import torch
from torchvision import models
import faiss
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from config import INDEX_FILE

# --- CACHED RESOURCES (AI Model Loading) ---

@st.cache_resource
def load_model_resnet():
    """Loads and caches the ResNet50 model for feature extraction."""
    with st.spinner("Loading ResNet50 model..."):
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Remove the final classification layer
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model.eval()
        return model

@st.cache_resource
def load_vlm():
    """Loads and caches the BLIP Vision-Language Model for dynamic tagging."""
    with st.spinner("Loading Vision-Language Model (BLIP) for dynamic tagging..."):
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        return processor, model

@st.cache_resource
def load_faiss_index(image_filenames):
    """Loads and caches the FAISS index."""
    if not os.path.exists(INDEX_FILE):
        return None, image_filenames
        
    with st.spinner("Loading FAISS Index..."):
        index = faiss.read_index(INDEX_FILE)
        return index, image_filenames