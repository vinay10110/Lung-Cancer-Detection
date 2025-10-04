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

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Open your browser**
   - The app will automatically open at `http://localhost:8501`

## 🔧 Environment Variables

The application uses environment variables for configuration. Copy `.env.example` to `.env` and modify as needed:

| Variable | Description | Default Value |
|----------|-------------|---------------|
| `MODEL_PATH` | Path to the trained model file | `./model2/conv2d-lung_detection-98.73.h5` |
| `PLOTS_DIR` | Directory containing training plots | `./plots & results` |

### Example .env file:
```env
# Model Configuration
MODEL_PATH=./model2/conv2d-lung_detection-98.73.h5

# Plots Directory
PLOTS_DIR=./plots & results
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
