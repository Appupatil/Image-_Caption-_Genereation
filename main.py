import streamlit as st
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import io
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="AI Image Caption Generator",
    page_icon="🖼️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .caption-box {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-top: 20px;
    }
    .caption-text {
        font-size: 1.3rem;
        font-weight: 500;
        color: #333;
    }
    .info-box {
        background-color: #fff9e6;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ffa500;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="main-header">🖼️ AI Image Caption Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload an image and let AI describe it for you!</div>', unsafe_allow_html=True)

# Cache the model loading to improve performance
@st.cache_resource
def load_model():
    """Load the pre-trained image captioning model"""
    with st.spinner("Loading AI model... This may take a moment on first run."):
        model_name = "nlpconnect/vit-gpt2-image-captioning"
        
        # Load model, processor, and tokenizer
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        image_processor = ViTImageProcessor.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        return model, image_processor, tokenizer, device

def generate_caption(image, model, image_processor, tokenizer, device, max_length=16, num_beams=4):
    """Generate caption for the input image"""
    # Preprocess image
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    
    # Generate caption
    with torch.no_grad():
        output_ids = model.generate(
            pixel_values,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
    
    # Decode the generated caption
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

# Load model
try:
    model, image_processor, tokenizer, device = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    model_loaded = False

# Main application layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image for best results"
    )
    
    # Advanced settings in an expander
    with st.expander("⚙️ Advanced Settings"):
        max_length = st.slider(
            "Maximum caption length",
            min_value=10,
            max_value=30,
            value=16,
            help="Maximum number of words in the caption"
        )
        num_beams = st.slider(
            "Beam search size",
            min_value=1,
            max_value=8,
            value=4,
            help="Higher values may produce better captions but take longer"
        )

with col2:
    st.markdown("### 🎯 Generated Caption")
    
    if uploaded_file is not None and model_loaded:
        try:
            # Display uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            
            # Display image with caption
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Generate caption button
            if st.button("✨ Generate Caption", type="primary", use_container_width=True):
                with st.spinner("🤖 AI is analyzing your image..."):
                    # Generate caption
                    caption = generate_caption(
                        image, model, image_processor, tokenizer, device,
                        max_length=max_length,
                        num_beams=num_beams
                    )
                    
                    # Display caption in a styled box
                    st.markdown(f"""
                        <div class="caption-box">
                            <p class="caption-text">📝 {caption.capitalize()}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Success message
                    st.success("✅ Caption generated successfully!")
                    
                    # Additional info
                    st.info(f"**Model used:** Vision Transformer (ViT) + GPT-2\n\n**Device:** {device.type.upper()}")
        
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
    
    elif uploaded_file is None:
        st.info("👈 Please upload an image to get started!")
    
    elif not model_loaded:
        st.error("❌ Model could not be loaded. Please refresh the page.")

# Information section at the bottom
st.markdown("---")
st.markdown("### 📚 How It Works")

col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.markdown("""
        **🔍 Image Analysis**
        
        The Vision Transformer (ViT) encodes the image into a meaningful representation by breaking it into patches and analyzing visual features.
    """)

with col_info2:
    st.markdown("""
        **🧠 Caption Generation**
        
        The GPT-2 decoder generates human-like text descriptions based on the encoded image features using advanced language modeling.
    """)

with col_info3:
    st.markdown("""
        **✨ Beam Search**
        
        Multiple caption candidates are generated and the best one is selected based on probability scores for optimal results.
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Built with ❤️ using Streamlit, PyTorch, and Hugging Face Transformers</p>
        <p><small>Model: nlpconnect/vit-gpt2-image-captioning</small></p>
    </div>
""", unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.markdown("### 🎯 About This App")
    st.info("""
        This application uses state-of-the-art deep learning to automatically generate 
        captions for images. The model combines:
        
        - **Vision Transformer (ViT)**: For image encoding
        - **GPT-2**: For text generation
        
        The model has been trained on millions of image-caption pairs and can describe 
        a wide variety of scenes, objects, and activities.
    """)
    
    st.markdown("### 💡 Tips for Best Results")
    st.markdown("""
        - Use clear, well-lit images
        - Avoid overly complex scenes
        - Images with distinct subjects work best
        - Try different beam search values for variety
    """)
    
    st.markdown("### 🚀 Model Details")
    st.markdown("""
        - **Architecture**: ViT-GPT2
        - **Parameters**: ~300M
        - **Training Data**: COCO Dataset
        - **Framework**: PyTorch + Transformers
    """)