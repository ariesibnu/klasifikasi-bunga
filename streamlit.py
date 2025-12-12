import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Set page config
st.set_page_config(
    page_title="Aplikasi Klasifikasi Bunga VGG16",
    page_icon="🌸",
    layout="wide"
)

# Title and description
st.title("Aplikasi Klasifikasi Bunga dengan VGG16")
st.markdown("""
Upload gambar bunga dan biarkan model AI memprediksi jenis bunganya!
""")

# Sidebar for additional info
with st.sidebar:
    st.header("About")
    st.markdown("""
    Aplikasi ini menggunakan model deep learning berbasis VGG16 yang dilatih pada 5 kategori bunga:
    - Daisy
    - Dandelion
    - Mawar
    - Bunga Matahari
    - Tulip
    
    Model ini dilatih menggunakan transfer learning dari bobot ImageNet.
    """)
    
    st.header("Instructions")
    st.markdown("""
    1. Unggah gambar bunga (JPEG/PNG)
    2. Aplikasi akan memproses dan menampilkan gambar Anda
    3. Klik 'Prediksi' untuk melihat hasil klasifikasi
    4. Lihat skor kepercayaan untuk setiap jenis bunga
    """)

# Load model with caching
@st.cache_resource
def load_vgg16_model():
    """Load the trained VGG16 model"""
    try:
        model = load_model('model_vgg16.h5')
        return model
    except:
        st.error("Model file 'model_vgg16.h5' not found. Please ensure the model file is in the same directory.")
        return None

# Class names (update according to your training data)
class_names = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']

# Function to preprocess image
def preprocess_image(img):
    """Preprocess image for VGG16 model"""
    # Resize image to 224x224
    img = img.resize((224, 224))
    
    # Convert image to array
    img_array = image.img_to_array(img)
    
    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess for VGG16
    img_array = preprocess_input(img_array)
    
    return img_array

# Function to make prediction
def predict_image(model, img_array):
    """Make prediction using the model"""
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    # Get all confidence scores
    confidence_scores = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
    
    return predicted_class, confidence, confidence_scores

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.header("Upload Image")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a flower image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image of a flower (daisy, dandelion, rose, sunflower, or tulip)"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image_pil = Image.open(uploaded_file)
        st.image(image_pil, caption="Uploaded Image", use_column_width=True)
        
        # Get image details
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.2f} KB",
            "Image format": image_pil.format,
            "Image size": f"{image_pil.size[0]} x {image_pil.size[1]} pixels"
        }
        
        with st.expander("Image Details"):
            for key, value in file_details.items():
                st.write(f"**{key}:** {value}")

with col2:
    st.header("Prediction Results")
    
    if uploaded_file is not None:
        # Load model
        model = load_vgg16_model()
        
        if model:
            # Create two columns for buttons
            btn_col1, btn_col2 = st.columns(2)
            
            with btn_col1:
                predict_button = st.button("🔮 Predict", type="primary", use_container_width=True)
            
            with btn_col2:
                clear_button = st.button("🔄 Clear", use_container_width=True)
            
            if predict_button:
                with st.spinner("Processing image and making prediction..."):
                    # Preprocess image
                    img_array = preprocess_image(image_pil)
                    
                    # Make prediction
                    predicted_class, confidence, confidence_scores = predict_image(model, img_array)
                    
                    # Display results
                    st.success(f"✅ **Prediction:** {class_names[predicted_class]}")
                    st.metric("Confidence", f"{confidence:.2%}")
                    
                    # Show confidence scores as a bar chart
                    st.subheader("Confidence Scores")
                    
                    # Sort confidence scores
                    sorted_scores = dict(sorted(confidence_scores.items(), 
                                                key=lambda item: item[1], 
                                                reverse=True))
                    
                    # Create bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = ['#FF6B6B' if i == predicted_class else '#4ECDC4' for i in range(len(class_names))]
                    ax.barh(list(sorted_scores.keys()), list(sorted_scores.values()), color=colors)
                    ax.set_xlabel('Confidence Score')
                    ax.set_title('Prediction Confidence by Class')
                    ax.set_xlim([0, 1])
                    
                    # Add value labels
                    for i, (key, val) in enumerate(sorted_scores.items()):
                        ax.text(val + 0.01, i, f'{val:.3f}', va='center')
                    
                    st.pyplot(fig)
                    
                    # Display as table
                    with st.expander("View Detailed Scores"):
                        for flower, score in sorted_scores.items():
                            st.write(f"**{flower}:** {score:.3%}")
                    
                    # Interpretation
                    st.info(f"""
                    **Interpretation:** The model is {confidence:.2%} confident that this is a **{class_names[predicted_class]}**.
                    """)
            
            if clear_button:
                st.rerun()
    else:
        st.info("👆 Please upload an image to get started!")
        
        # Placeholder for results
        st.markdown("""
        ### Example Results:
        
        After uploading an image and clicking **Predict**, you'll see:
        1. **Predicted flower type**
        2. **Confidence score**
        3. **Detailed confidence scores** for all flower types
        4. **Visual chart** of predictions
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit and TensorFlow | VGG16 Transfer Learning Model</p>
</div>
""", unsafe_allow_html=True)

# Add some custom CSS
st.markdown("""
<style>
    .stButton button {
        width: 100%;
        border-radius: 10px;
        font-weight: bold;
        font-size: 16px;
        height: 50px;
    }
    
    .stFileUploader {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 20px;
    }
    
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    
    .css-1aumxhk {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)