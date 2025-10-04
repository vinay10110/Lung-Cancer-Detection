import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import os
from PIL import Image
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Lung Cancer Detection System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        text-align: center;
        margin-bottom: 1rem;
    }
    .cancer-detected {
        background-color: #ffebee;
        border: 2px solid #f44336;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    }
    .no-cancer {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    }
    .prediction-box {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    }
    .stProgress .st-bo {
        background-color: #2E86AB;
    }
</style>
""", unsafe_allow_html=True)

# Cache the model loading to improve performance
@st.cache_resource
def load_lung_cancer_model():
    """Load the lung cancer detection model."""
    # Get model path from environment variable with fallback to default
    MODEL_PATH = os.getenv("MODEL_PATH", "./model2/conv2d-lung_detection-98.73.h5")
    
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at {MODEL_PATH}")
        st.error("💡 Make sure to set the MODEL_PATH environment variable or place the model in the default location.")
        st.stop()
    
    try:
        with st.spinner("Loading AI model..."):
            model = load_model(MODEL_PATH)
        st.success(f"✅ AI Model loaded successfully from: {MODEL_PATH}")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# Model and class definitions
model = load_lung_cancer_model()

CLASS_LABELS = [
    "Lung Adenocarcinoma",
    "Lung Benign Tissue", 
    "Lung Squamous Cell Carcinoma"
]

CANCER_STATUS = {
    "Lung Adenocarcinoma": "CANCER DETECTED",
    "Lung Benign Tissue": "NO CANCER",
    "Lung Squamous Cell Carcinoma": "CANCER DETECTED"
}

CLASS_DESCRIPTIONS = {
    "Lung Adenocarcinoma": "A type of non-small cell lung cancer that typically develops in the outer areas of the lungs.",
    "Lung Benign Tissue": "Normal, healthy lung tissue with no signs of malignancy.",
    "Lung Squamous Cell Carcinoma": "A type of non-small cell lung cancer that usually develops in the central part of the lungs."
}

def preprocess_image_for_prediction(uploaded_file):
    """Preprocess uploaded image for prediction."""
    try:
        # Open image using PIL
        img = Image.open(uploaded_file)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to model input size
        img = img.resize((224, 224))
        
        # Convert to array
        img_array = np.array(img)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, img
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        return None, None

def predict_lung_cancer_streamlit(img_array):
    """Make prediction using the loaded model."""
    try:
        # Make prediction
        with st.spinner("🔍 Analyzing image..."):
            predictions = model.predict(img_array, verbose=0)
        
        # Get results
        predicted_class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        predicted_class = CLASS_LABELS[predicted_class_idx]
        cancer_status = CANCER_STATUS[predicted_class]
        
        return predicted_class, confidence, cancer_status, predictions[0]
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        return None, None, None, None

def display_results(predicted_class, confidence, cancer_status, all_predictions, processed_img):
    """Display prediction results in a formatted way."""
    
    # Main result display
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Analysis Results")
        
        # Color-coded result box
        if "CANCER" in cancer_status and cancer_status != "NO CANCER":
            st.markdown(f"""
            <div class="cancer-detected">
                <h3 style="color: #d32f2f; margin: 0;">⚠️ {cancer_status}</h3>
                <p style="margin: 10px 0; color: #333;"><strong>Detected Condition:</strong> {predicted_class}</p>
                <p style="margin: 10px 0; color: #333;"><strong>Confidence Level:</strong> {confidence:.2%}</p>
                <p style="margin: 10px 0; font-size: 0.9em; color: #666;">
                    {CLASS_DESCRIPTIONS[predicted_class]}
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="no-cancer">
                <h3 style="color: #2e7d32; margin: 0;">✅ {cancer_status}</h3>
                <p style="margin: 10px 0; color: #333;"><strong>Tissue Type:</strong> {predicted_class}</p>
                <p style="margin: 10px 0; color: #333;"><strong>Confidence Level:</strong> {confidence:.2%}</p>
                <p style="margin: 10px 0; font-size: 0.9em; color: #666;">
                    {CLASS_DESCRIPTIONS[predicted_class]}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("🖼️ Analyzed Image")
        st.image(processed_img, caption="Processed Image (224x224)", use_column_width=True)
    
    # Detailed predictions
    st.subheader("📊 Detailed Analysis")
    
    # Create probability bars
    for i, (class_name, prob) in enumerate(zip(CLASS_LABELS, all_predictions)):
        status = CANCER_STATUS[class_name]
        
        # Color coding for different classes
        if "CANCER" in status and status != "NO CANCER":
            color = "#f44336"  # Red for cancer
        else:
            color = "#4caf50"  # Green for benign
        
        # Progress bar with custom styling
        st.write(f"**{class_name}** ({status})")
        st.progress(float(prob))
        st.write(f"Probability: {prob:.4f} ({prob*100:.2f}%)")
        st.write("---")

def prediction_page():
    """Image prediction page."""
    st.header("📤 Upload Lung Tissue Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload a lung tissue image for analysis"
    )
    
    if uploaded_file is not None:
        # Display uploaded image info
        st.success(f"✅ Image uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")
        
        # Preprocess image
        img_array, processed_img = preprocess_image_for_prediction(uploaded_file)
        
        if img_array is not None:
            # Make prediction
            predicted_class, confidence, cancer_status, all_predictions = predict_lung_cancer_streamlit(img_array)
            
            if predicted_class is not None:
                # Display results
                display_results(predicted_class, confidence, cancer_status, all_predictions, processed_img)
                
                # Additional information
                st.header("🔬 Technical Details")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Image Size", "224×224 pixels")
                with col2:
                    st.metric("Color Mode", "RGB")
                with col3:
                    st.metric("Model Confidence", f"{confidence:.2%}")
                
                # Download results option
                st.header("💾 Export Results")
                
                results_text = f"""
Lung Cancer Detection Results
============================
Image: {uploaded_file.name}
Analysis Date: {st.session_state.get('analysis_date', 'N/A')}

RESULTS:
- Predicted Class: {predicted_class}
- Cancer Status: {cancer_status}
- Confidence: {confidence:.4f} ({confidence*100:.2f}%)

DETAILED PREDICTIONS:
"""
                for class_name, prob in zip(CLASS_LABELS, all_predictions):
                    status = CANCER_STATUS[class_name]
                    results_text += f"- {class_name}: {prob:.4f} ({prob*100:.2f}%) - {status}\n"
                
                results_text += """
DISCLAIMER:
This analysis is for educational/research purposes only.
Always consult qualified healthcare professionals for medical diagnosis.
"""
                
                st.download_button(
                    label="📄 Download Results as Text",
                    data=results_text,
                    file_name=f"lung_cancer_analysis_{uploaded_file.name}.txt",
                    mime="text/plain"
                )
    
    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload a lung tissue image to begin analysis")

def model_info_page():
    """Model information and training details page."""
    st.header("🤖 Model Information & Training Details")
    
    # Model Overview
    st.subheader("📊 Model Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Model Architecture:**
        - **Type**: Convolutional Neural Network (CNN)
        - **Model File**: {os.getenv("MODEL_PATH", "./model2/conv2d-lung_detection-98.73.h5")}
        - **Input Size**: 224 × 224 × 3 (RGB)
        - **Framework**: TensorFlow/Keras
        - **Classes**: 3 lung tissue types
        """)
        
        st.markdown("""
        **Training Configuration:**
        - **Dataset**: Lung tissue histopathological images
        - **Image Preprocessing**: Resize to 224×224, RGB normalization
        - **Validation Split**: Standard train/validation split
        - **Optimization**: Adam optimizer
        """)
    
    with col2:
        st.markdown("""
        **Model Performance:**
        - **Overall Accuracy**: 98.73%
        - **Model Size**: ~254 MB
        - **Inference Time**: < 1 second per image
        - **Color Mode**: RGB (3 channels)
        """)
        
        st.markdown("""
        **Key Features:**
        - High accuracy lung cancer detection
        - Multi-class classification capability
        - Robust to various image qualities
        - Optimized for medical imaging
        """)
    
    # Training Metrics and Plots
    st.subheader("📈 Training Metrics & Visualizations")
    
    # Display training plots if available
    plots_dir = os.getenv("PLOTS_DIR", "./plots & results")
    if os.path.exists(plots_dir):
        plot_files = [f for f in os.listdir(plots_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if plot_files:
            st.write("**Training Performance Visualizations:**")
            
            # Create columns for plots
            if len(plot_files) >= 2:
                col1, col2 = st.columns(2)
                
                # Training history plot
                if "training_history.png" in plot_files:
                    with col1:
                        st.write("**Training History**")
                        training_plot_path = os.path.join(plots_dir, "training_history.png")
                        st.image(training_plot_path, caption="Model Training History - Loss and Accuracy over Epochs")
                
                # Confusion matrix
                if "confusion_matrix.png" in plot_files:
                    with col2:
                        st.write("**Confusion Matrix**")
                        confusion_plot_path = os.path.join(plots_dir, "confusion_matrix.png")
                        st.image(confusion_plot_path, caption="Confusion Matrix - Model Performance on Test Set")
            
            # Display any additional plots
            other_plots = [f for f in plot_files if f not in ["training_history.png", "confusion_matrix.png"]]
            if other_plots:
                st.write("**Additional Visualizations:**")
                for plot_file in other_plots:
                    plot_path = os.path.join(plots_dir, plot_file)
                    st.image(plot_path, caption=f"{plot_file}")
        else:
            st.info("No training plots found in the plots directory.")
    else:
        st.warning("Plots directory not found.")
    
    # Model Architecture Details
    st.subheader("🏗️ Model Architecture")
    
    st.markdown("""
    **Convolutional Neural Network Details:**
    
    The model uses a deep convolutional neural network architecture specifically designed for medical image classification:
    
    1. **Input Layer**: 224×224×3 RGB images
    2. **Convolutional Layers**: Multiple conv2d layers with ReLU activation
    3. **Pooling Layers**: MaxPooling for feature reduction
    4. **Dropout Layers**: Regularization to prevent overfitting
    5. **Dense Layers**: Fully connected layers for classification
    6. **Output Layer**: 3-class softmax for lung tissue classification
    
    **Training Process:**
    - **Data Augmentation**: Applied to increase dataset diversity
    - **Early Stopping**: Prevented overfitting during training
    - **Learning Rate Scheduling**: Optimized convergence
    - **Cross-Validation**: Ensured robust performance metrics
    """)
    
    # Technical Specifications
    st.subheader("⚙️ Technical Specifications")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Accuracy", "98.73%")
        st.metric("Input Resolution", "224×224")
    
    with col2:
        st.metric("Number of Classes", "3")
        st.metric("Model Size", "254 MB")
    
    with col3:
        st.metric("Color Channels", "3 (RGB)")
        st.metric("Framework", "TensorFlow")

def classes_info_page():
    """Lung cancer classes information page."""
    st.header("🫁 Lung Cancer Classes Information")
    
    st.write("""
    This section provides detailed information about the three lung tissue classes that our AI model can detect and classify.
    """)
    
    # Class 1: Lung Adenocarcinoma
    st.subheader("🔴 Class 1: Lung Adenocarcinoma")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Medical Definition:**
        Lung adenocarcinoma is the most common type of lung cancer, accounting for about 40% of all lung cancer cases. 
        It is a type of non-small cell lung cancer (NSCLC).
        
        **Characteristics:**
        - **Location**: Usually develops in the outer areas of the lungs (peripheral)
        - **Growth Pattern**: Tends to grow more slowly than other lung cancers
        - **Cell Type**: Originates from mucus-producing cells in the lungs
        - **Smoking Relation**: Can occur in both smokers and non-smokers
        
        **Clinical Features:**
        - Often asymptomatic in early stages
        - May present with persistent cough, shortness of breath
        - Can metastasize to lymph nodes and distant organs
        - Better prognosis when detected early
        """)
    
    with col2:
        st.markdown("""
        **Key Statistics:**
        - **Prevalence**: ~40% of lung cancers
        - **Gender**: More common in women
        - **Age**: Typically 60+ years
        - **5-year Survival**: 15-20% overall
        
        **Risk Factors:**
        - Smoking (primary)
        - Secondhand smoke
        - Radon exposure
        - Air pollution
        - Genetic factors
        """)
    
    # Class 2: Lung Benign Tissue
    st.subheader("🟢 Class 2: Lung Benign Tissue")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Medical Definition:**
        Benign lung tissue represents normal, healthy lung parenchyma without any malignant changes. 
        This tissue maintains normal cellular architecture and function.
        
        **Characteristics:**
        - **Structure**: Normal alveolar and bronchial architecture
        - **Cell Morphology**: Regular cell shapes and sizes
        - **Growth Pattern**: No abnormal proliferation
        - **Function**: Normal gas exchange capability
        
        **Histological Features:**
        - Well-organized tissue structure
        - Normal cell-to-cell relationships
        - Absence of atypical cellular features
        - Proper tissue boundaries and organization
        """)
    
    with col2:
        st.markdown("""
        **Identification Markers:**
        - **Cell Size**: Uniform and normal
        - **Nuclear Features**: Regular nuclei
        - **Tissue Organization**: Well-structured
        - **Cellular Density**: Normal distribution
        
        **Clinical Significance:**
        - Indicates healthy lung function
        - No treatment required
        - Regular monitoring recommended
        - Baseline for comparison
        """)
    
    # Class 3: Lung Squamous Cell Carcinoma
    st.subheader("🟠 Class 3: Lung Squamous Cell Carcinoma")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Medical Definition:**
        Lung squamous cell carcinoma is another type of non-small cell lung cancer, accounting for about 25-30% 
        of all lung cancer cases. It develops from the flat cells lining the airways.
        
        **Characteristics:**
        - **Location**: Usually develops in the central part of the lungs (near bronchi)
        - **Cell Origin**: Arises from squamous epithelial cells
        - **Growth Pattern**: Can grow rapidly and spread to nearby tissues
        - **Smoking Relation**: Strongly associated with smoking history
        
        **Clinical Features:**
        - Often presents with cough and hemoptysis (blood in sputum)
        - May cause airway obstruction
        - Can lead to pneumonia or lung collapse
        - Tends to remain localized longer than adenocarcinoma
        """)
    
    with col2:
        st.markdown("""
        **Key Statistics:**
        - **Prevalence**: ~25-30% of lung cancers
        - **Gender**: More common in men
        - **Age**: Typically 60+ years
        - **5-year Survival**: 15-25% overall
        
        **Risk Factors:**
        - Heavy smoking (strongest link)
        - Long-term smoking history
        - Occupational exposures
        - Air pollution
        - Previous lung disease
        """)
    
    # Comparison Table
    st.subheader("📊 Class Comparison Summary")
    
    comparison_data = {
        "Feature": [
            "Cancer Type",
            "Location in Lung",
            "Smoking Association",
            "Growth Rate",
            "Common Symptoms",
            "Prognosis",
            "Treatment Approach"
        ],
        "Lung Adenocarcinoma": [
            "Malignant (Cancer)",
            "Peripheral/Outer areas",
            "Moderate (smokers & non-smokers)",
            "Moderate",
            "Often asymptomatic early",
            "Variable, better if early",
            "Surgery, chemo, targeted therapy"
        ],
        "Lung Benign Tissue": [
            "Normal/Healthy",
            "Throughout lung",
            "None",
            "None (normal)",
            "None",
            "Excellent",
            "No treatment needed"
        ],
        "Lung Squamous Cell Carcinoma": [
            "Malignant (Cancer)",
            "Central/Near bronchi",
            "Very strong",
            "Can be rapid",
            "Cough, blood in sputum",
            "Variable, depends on stage",
            "Surgery, radiation, chemotherapy"
        ]
    }
    
    df = pd.DataFrame(comparison_data)
    st.table(df)
    
    # Important Medical Disclaimer
    st.subheader("⚠️ Important Medical Information")
    
    st.warning("""
    **Medical Disclaimer:**
    
    This information is provided for educational purposes only and should not be used for medical diagnosis or treatment decisions. 
    
    **Key Points:**
    - This AI system is a research/educational tool
    - Results should always be confirmed by qualified medical professionals
    - Histopathological diagnosis requires expert pathologist review
    - Treatment decisions should involve oncology specialists
    - Early detection and professional medical care are crucial for lung cancer
    
    **If you have concerns about lung health, please consult with healthcare professionals immediately.**
    """)

def main():
    """Main Streamlit application with navigation."""
    
    # Header
    st.markdown('<h1 class="main-header">🫁 Lung Cancer Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Medical Image Analysis</p>', unsafe_allow_html=True)
    
    # Sidebar Navigation
    with st.sidebar:
        st.title("🧭 Navigation")
        
        # Navigation options
        page = st.selectbox(
            "Choose a section:",
            ["🔍 Predict Images", "🤖 Model Information", "🫁 Lung Cancer Classes"],
            index=0
        )
        
        st.markdown("---")
        
        # Quick info based on current page
        if page == "🔍 Predict Images":
            st.header("📋 Prediction Guide")
            st.write("""
            **Steps:**
            1. Upload a lung tissue image
            2. Wait for AI analysis
            3. Review the results
            4. Download results if needed
            
            **Supported Formats:**
            JPG, JPEG, PNG, BMP, TIFF
            """)
            
        elif page == "🤖 Model Information":
            st.header("ℹ️ Model Overview")
            st.write("""
            **Key Metrics:**
            - Accuracy: 98.73%
            - Classes: 3 lung tissue types
            - Model: CNN (TensorFlow)
            - Input: 224×224 RGB images
            """)
            
        elif page == "🫁 Lung Cancer Classes":
            st.header("📚 Class Overview")
            st.write("""
            **Three Classes:**
            1. Lung Adenocarcinoma (Cancer)
            2. Lung Benign Tissue (Normal)
            3. Lung Squamous Cell Carcinoma (Cancer)
            """)
        
        st.markdown("---")
        
        # General disclaimer
        st.header("⚠️ Important Notice")
        st.write("""
        This tool is for educational/research purposes only. 
        Always consult healthcare professionals for medical diagnosis.
        """)
    
    # Main content based on navigation
    if page == "🔍 Predict Images":
        prediction_page()
    elif page == "🤖 Model Information":
        model_info_page()
    elif page == "🫁 Lung Cancer Classes":
        classes_info_page()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8em;">
        <p>🤖 Powered by TensorFlow & Streamlit | 🫁 Lung Cancer Detection System</p>
        <p>⚠️ For educational and research purposes only. Not for clinical diagnosis.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
