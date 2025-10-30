# Vision-Cop

Vision COP - Image Authenticity & Source Engine

A Deep Learning & High-Speed Indexing Solution for Image Verification

This project implements a scalable engine for validating image authenticity and tracing original sources using Content-Based Image Retrieval (CBIR). The core search engine is designed as a high-performance FastAPI service, making it ready for cloud deployment on platforms like AWS or GCP.

# 🌟 Key Features

- Deep Feature Extraction: Uses a pre-trained ResNet50 model (PyTorch) to generate robust 2048-dimensional feature vectors (embeddings).

- High-Speed Search: Employs FAISS (Facebook AI Similarity Search) for lightning-fast nearest-neighbor search across a large database of indexed image vectors.

- Production-Ready API: The core search logic is encapsulated in an asynchronous FastAPI service (api.py), enabling high concurrency and cloud scalability.

- Authenticity Checks: Integrates Perceptual Hashing (pHash) to detect minor manipulations, resizes, or simple crops/rotations on potential source matches.

- Vision-Language Context: Utilizes the BLIP Vision-Language Model (VLM) for automatic image tagging and context verification against user-provided claims.

- Streamlit Frontend: An interactive user interface built with Streamlit (app.py) serves as the client, making requests to the API for search and analysis.



# 🚀 Getting Started (Local Development)

Follow these steps to set up and run the project locally.

- Prerequisites:

Ensure you have Python 3.8+ installed.


- Installation:

Clone the repository and install all required libraries. This installation includes PyTorch, FAISS-CPU, HuggingFace Transformers, and the necessary web frameworks.

```
# Clone the repository
git clone [YOUR_REPOSITORY_URL_HERE]
cd [YOUR_PROJECT_FOLDER]

# Install all dependencies
pip install torch torchvision torchaudio faiss-cpu streamlit pillow numpy tqdm transformers imagehash fastapi uvicorn
```

- Dataset & Index Setup:

The system relies on a pre-computed FAISS index.

Create a folder named ```images``` in the root directory and place all your reference images (verified sources) inside it.

Run the index creation script (e.g., ```python create_index.py```) to generate the ```faiss_index.bin``` and ```faiss_index.pkl``` files.


- Running the Application:

For a full local test, you would need to run the API and the Streamlit frontend concurrently.

A. Start the Backend API (FastAPI)

Run the API using Uvicorn on a specific port (e.g., 8000).
```
uvicorn api:app --reload --host 0.0.0.0 --port 8000

```
Note: Ensure the API_URL in your Streamlit app points to this local address (```http://localhost:8000/search/```).

B. Start the Frontend UI (Streamlit)

In a separate terminal window, start the Streamlit app.
```
streamlit run app.py

```
# ⚙️ Architecture and Deployment

- Separation of Concerns

The project follows a microservice pattern for optimal scalability:

Frontend (```app.py```): Handles image upload and visualization. Client role.

Backend (```api.py```): Handles model loading, feature extraction, and high-speed search. Server role.

- Cloud Deployment Readiness

The ```api.py``` file is designed for cloud deployment:

It uses the ```@app.on_event("startup")``` hook to load memory-intensive models (ResNet50, FAISS Index) only once, which is the standard practice for cloud container services (e.g., AWS ECS/Fargate or GCP Cloud Run).

The use of FastAPI enables asynchronous request handling, allowing a single deployed instance to manage many simultaneous search requests efficiently.

Checks semantic context (BLIP VLM) against user claims.

Flags inconsistencies, such as claiming a disaster when AI tags suggest a peaceful scene.
