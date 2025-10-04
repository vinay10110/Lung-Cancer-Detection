# 🫁 Lung Cancer Detection System

AI-Powered Medical Image Analysis for Lung Cancer Detection using Deep Learning.

## 🚀 Features

- **AI-Powered Detection**: Uses a Convolutional Neural Network with 98.73% accuracy
- **Multi-Class Classification**: Detects 3 types of lung tissue conditions
- **Interactive Web Interface**: User-friendly Streamlit application
- **Educational Content**: Comprehensive information about lung cancer types
- **Model Transparency**: Detailed training metrics and visualizations

## 🛠️ Tech Stack

- **Backend**: Python, TensorFlow/Keras
- **Frontend**: Streamlit
- **Image Processing**: PIL, NumPy
- **Data Visualization**: Matplotlib, Pandas
- **Configuration**: python-dotenv

## 🏃‍♂️ Run Locally

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/vinay10110/Lung-Cancer-Detection.git
   cd Lung-Cancer-Detection
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env file with your configurations
   # The default values should work for most setups
   ```

5. **Download Model Weights**
   
   You need to download the trained model weights before running the application:
   
   **Option 1: Lung Cancer Only Model (Recommended for this app)**
   - Download from: [Lung Cancer Detection Weights](https://mega.nz/folder/EqInzIQS#RYUe5FHArGuBJB23WTPNVQ)
   - Extract and place the model file in `./model2/` directory
   
   **Option 2: Combined Lung + Colon Cancer Model**
   - Download from: [Lung + Colon Cancer Weights](https://mega.nz/folder/xzBDyTaa#Dc1UBX1X00qrymgM2Axjmw)
   - Extract and place the model file in `./model/` directory
   - Update your `.env` file with the correct `MODEL_PATH`

6. **Run the application**
   ```bash
   streamlit run app.py
   ```

7. **Open your browser**
   - The app will automatically open at `http://localhost:8501`

## 🔧 Environment Variables

The application uses environment variables for configuration. Copy `.env.example` to `.env` and modify as needed:

| Variable | Description | Default Value |
|----------|-------------|---------------|
| `MODEL_PATH` | Path to the trained model file | `./model2/conv2d-lung_detection-98.73.h5` |

### Example .env file:
```env
# Model Configuration
MODEL_PATH=./model2/conv2d-lung_detection-98.73.h5
```

## 📥 Model Weights Download

### Available Models

**🫁 Lung Cancer Only Model (Recommended)**
- **Download Link**: [Lung Cancer Detection Weights](https://mega.nz/folder/EqInzIQS#RYUe5FHArGuBJB23WTPNVQ)
- **Classes**: 3 lung tissue types
- **Accuracy**: 98.73%
- **File Location**: Place in `./model2/` directory
- **Best For**: This Streamlit application (lung cancer detection only)

**🫁🦠 Combined Lung + Colon Cancer Model**
- **Download Link**: [Lung + Colon Cancer Weights](https://mega.nz/folder/xzBDyTaa#Dc1UBX1X00qrymgM2Axjmw)
- **Classes**: 5 tissue types (lung + colon)
- **File Location**: Place in `./model/` directory
- **Best For**: Research projects requiring both lung and colon cancer detection
- **Note**: Requires updating `MODEL_PATH` in `.env` file

### Setup Instructions

1. **Choose your model** based on your needs
2. **Download** from the appropriate MEGA link
3. **Extract** the downloaded files
4. **Place** the model file in the correct directory:
   - Lung only: `./model2/conv2d-lung_detection-98.73.h5`
   - Lung + Colon: `./model/conv2d-lung_cancer_detection-99.68.h5`
5. **Update** your `.env` file if using the combined model:
   ```env
   MODEL_PATH=./model/conv2d-lung_cancer_detection-99.68.h5
   ```

## 📊 Key Results & Visualizations

- **Model Accuracy**: 98.73%
- **Classes**: 3 lung tissue types
- **Training Visualizations**: Available in the Model Information section
- **Confusion Matrix**: Shows model performance across all classes

## 🏥 Lung Cancer Classes

1. **Lung Adenocarcinoma** - Most common type of lung cancer (40% of cases)
2. **Lung Benign Tissue** - Normal, healthy lung tissue
3. **Lung Squamous Cell Carcinoma** - Cancer type strongly associated with smoking (25-30% of cases)

## ⚠️ Important Disclaimer

This tool is for **educational and research purposes only**. 

- Results should always be confirmed by qualified medical professionals
- Not intended for clinical diagnosis or treatment decisions
- Always consult healthcare professionals for medical concerns

## 🤝 Acknowledgements

This project was developed as part of medical AI research and education initiatives.

---

**🤖 Powered by TensorFlow & Streamlit | 🫁 Lung Cancer Detection System**
