# app.py - Main Streamlit UI/Frontend

import streamlit as st
from PIL import Image
import os

# Import everything from the new modular files
from config import ACCENTURE_PURPLE, TOP_K, DATASET_PATH, load_css
from models import load_model_resnet, load_vlm, load_faiss_index
from utils import (
    get_image_filenames, extract_features, search_index, 
    verify_phash, generate_tags, check_metadata_mismatch
)

def main():
    st.set_page_config(page_title="Image Authenticity Engine", layout="wide")
    
    # Load the CSS (including Glassmorphism)
    load_css()
    
    st.markdown('<p class="big-font">VISION COP</p>', unsafe_allow_html=True)
    st.markdown('<p class="small-font">Image Source and Authenticity Engine</p>', unsafe_allow_html=True)
    st.markdown("---")

    # --- Initial Load of Models and Index ---
    image_filenames = get_image_filenames()
    model_resnet = load_model_resnet()
    processor_blip, model_blip = load_vlm() 
    faiss_index, image_ids = load_faiss_index(image_filenames)
    
    if faiss_index is None:
        st.error(f"FATAL: Index file not found. Please run 'create_index.py' first!")
        st.stop() 

    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.header("Query Image and Claim")
        
        uploaded_file = st.file_uploader("Upload Image to Verify", type=["jpg", "jpeg", "png"])
        
        st.markdown("---")
        st.subheader("Simulated News/Social Media Claim")
        
        claim_tag = st.text_input("Headline/Claim Keywords", value="Terrorist Attack in London 2025")
        claim_location = st.text_input("Claimed Location", value="London, UK")
        claim_year = st.number_input("Claimed Year", min_value=1990, max_value=2025, value=2025)

        query_context = {"tag": claim_tag, "location": claim_location, "year": claim_year}

    with col2:
        st.header("Engine Results")
        
        if uploaded_file is not None:
            query_img = Image.open(uploaded_file)
            st.image(query_img, caption=f"Query Image: {uploaded_file.name}", use_container_width=True)
            
            # --- RUN BUTTON ---
            if st.button("Run Authenticity Check :mag:", type="primary"):
                
                # --- START: FEATURE EXECUTION ---
                p_distance, p_distance_flipped, generated_tags, generated_caption, original_path = (None,) * 5

                with st.spinner("Running All Checks..."):
                    # 1. CBIR Search
                    query_features = extract_features(query_img, model_resnet)
                    search_results = search_index(faiss_index, query_features, image_ids, TOP_K)
                    
                    if search_results:
                        top_match = search_results[0]
                        original_path = os.path.join(DATASET_PATH, top_match['filename'])
                        
                        # 2. pHash Check
                        p_distance, p_distance_flipped = verify_phash(query_img, original_path)
                        
                        # 3. VLM Context Check
                        generated_tags, generated_caption = generate_tags(original_path, processor_blip, model_blip)
                        
                        original_source_data = {
                            "location": "Source: BLIP VLM (Auto-Generated)",
                            "year": 2020, 
                            "tags": generated_tags
                        }
                        mismatch_flags = check_metadata_mismatch(query_context, original_source_data)
                
                # --- END: FEATURE EXECUTION ---
                
                st.markdown("---")
                
                tab1, tab2, tab3 = st.tabs(["1. Source & Similarity (CBIR)", "2. Image Integrity (pHash)", "3. Context Mismatch (VLM)"])

                if search_results:
                    
                    # --- TAB 1: SOURCE & SIMILARITY (CBIR) ---
                    with tab1:
                        st.subheader("AI-Powered Reverse Image Lookup :camera:")
                        
                        if search_results[0]['similarity_percent'] < 50:
                            st.warning(f"Low Confidence: Top match similarity is {search_results[0]['similarity_percent']:.1f}%. The image is likely new or heavily altered.")
                        else:
                            st.success(f"High Confidence: Found **{top_match['filename']}** with **{top_match['similarity_percent']:.1f}%** visual similarity.")

                        # Display Top K Results
                        cols = st.columns(TOP_K)
                        for i, result in enumerate(search_results):
                            original_path_display = os.path.join(DATASET_PATH, result['filename'])
                            with cols[i]:
                                st.image(original_path_display, caption=f"Sim: {result['similarity_percent']:.1f}%", use_container_width=True)


                    # --- TAB 2: IMAGE INTEGRITY (pHash) ---
                    with tab2:
                        st.subheader("Visual Integrity Check (Perceptual Hash)")
                        
                        if p_distance == -1:
                            st.info("pHash could not be calculated.")
                            
                        else:
                            # Integrity Gauge
                            integrity_score = 100 - (min(p_distance, 64) / 64) * 100
                            st.caption(f"Hamming Distance: {p_distance} / 64")
                            st.progress(integrity_score / 100, text=f"**Structural Integrity Score: {integrity_score:.1f}%**")

                            # Mirroring Detection
                            is_mirrored = (p_distance > 15) and (p_distance_flipped < 5) and (p_distance - p_distance_flipped > 10)
                            
                            if is_mirrored:
                                st.error(f"**VERDICT: MIRRORING/FLIPPING DETECTED** (Distance to Flipped: {p_distance_flipped}).")
                                st.info("This indicates the image has been structurally altered via horizontal reflection.")
                            
                            # General Integrity Check
                            elif p_distance <= 4: 
                                st.success(f"**VERDICT: HIGH INTEGRITY** (Minimal changes, e.g., minor compression).")
                            elif p_distance <= 15:
                                st.warning(f"**VERDICT: POTENTIAL MANIPULATION** (Minor structural changes found).")
                            else:
                                st.error(f"**VERDICT: SIGNIFICANT ALTERATION** (Heavy cropping or object removal detected).")


                    # --- TAB 3: CONTEXT MISMATCH (VLM) ---
                    with tab3:
                        st.subheader("Dynamic Context Verification (BLIP VLM) :clipboard:")
                        st.markdown(f"**News Claim Context:** **'{claim_tag}'** (Year: {claim_year}, Location: {claim_location})")
                        
                        st.markdown("---")
                        st.caption("Auto-Generated Source Context:")
                        st.info(f"VLM Caption: *{generated_caption}*")
                        
                        if mismatch_flags:
                            st.error("🚨 **VERDICT: DECEPTIVE VISUALS DETECTED** (Context Mismatch):")
                            for flag in mismatch_flags:
                                st.write(f"- {flag}")
                        else:
                            st.success("✅ **VERDICT: CONTEXT CONSISTENT** (Claimed details align with VLM analysis).")

                else:
                    st.error("No close visual matches found in the database. Cannot perform verification checks.")

if __name__ == '__main__':
    main()
