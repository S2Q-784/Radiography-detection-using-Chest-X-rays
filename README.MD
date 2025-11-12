# COVID-19 Radiography Detection System

A deep learning-based web application for automated classification of chest X-ray images into Normal, Pneumonia, and COVID-19 categories using Convolutional Neural Networks (CNN).

## Project Overview

Chest-related diseases like Pneumonia and COVID-19 pose serious public health concerns, especially when not detected at an early stage. Traditional diagnosis using chest X-rays requires experienced radiologists and can be time-consuming and prone to human error, especially in rural or under-resourced settings. 

This project implements an automated, reliable, and efficient radiographic detection system that analyzes chest X-ray images and provides instant classification, thereby aiding in faster diagnosis and clinical decision-making.

### Key Highlights

- **Multi-class Classification**: Distinguishes between Normal, Pneumonia, and COVID-19 cases
- **Custom CNN Architecture**: Optimized for medical image analysis
- **Real-time Predictions**: Flask-based web interface for instant results
- **High Accuracy**: Validated using comprehensive performance metrics
- **Local Deployment**: Ready-to-use application for healthcare facilities

## Objectives

- Develop a deep learning model that can automatically classify chest X-ray images
- Accurately distinguish between Normal, Pneumonia-infected, and COVID-19-infected lungs using radiographic data
- Reduce diagnostic time and assist healthcare professionals with a supportive diagnostic tool
- Evaluate model performance using standard metrics (accuracy, precision, recall, F1-score)
- Deploy a user-friendly interface for clinicians to upload X-ray images and receive instant predictions

## Scope

- The system focuses solely on **analysis of 2D chest X-ray images**
- Intended as a **decision-support tool** for preliminary screening, not a standalone diagnostic system
- Covers three specified classes: **Normal, Pneumonia, and COVID-19**
- Does not diagnose other chest-related diseases (e.g., tuberculosis, lung cancer)

## Project Structure

```
Radiography-detection-using-Chest-X-rays/
├── app.py                          # Main Flask application
├── ml_project[1].ipynb            # Jupyter notebook (model training & evaluation)
├── CNN_Covid19_Xray_Version.h5    # Trained CNN model
├── Label_encoder.pkl              # Label encoder for class mapping
├── templates/                    # HTML templates
│   ├── index.html               # Home page with upload interface
│   ├── result.html              # Results display page
│   └── camera.html              # Webcam capture page
└── uploads/                      # Directory for uploaded images
```

## Features

- **Multi-Class Classification**: Classifies X-rays into Normal, Pneumonia, or COVID-19
- **Image Upload Interface**: Easy-to-use file upload system
- **Webcam Integration**: Capture X-ray images directly through webcam
- **Real-time Prediction**: Instant classification with confidence scores
- **Visual Results Display**: Shows uploaded images alongside predictions
- **Robust Model Architecture**: Custom CNN with BatchNormalization and Dropout
- **Early Stopping**: Prevents overfitting during training

## Model Architecture

### CNN Design

The custom CNN architecture includes:

- **Convolutional Layers (Conv2D)**: Feature extraction from X-ray images
- **Max Pooling Layers (MaxPooling2D)**: Spatial dimension reduction
- **Batch Normalization**: Stabilizes learning process
- **Dropout Layers**: Prevents overfitting
- **Flatten Layer**: Converts 2D features to 1D
- **Dense Layers**: Fully connected layers for classification
- **Softmax Activation**: Multi-class probability distribution output

### Model Specifications

- **Input Size**: 150x150 RGB images
- **Normalization**: Pixel values scaled to [0, 1]
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Training Strategy**: EarlyStopping on validation loss
- **Classes**: 3 (Normal, Pneumonia, COVID-19)

## Model Performance

### Classification Metrics

The model demonstrates strong performance across all three classes:

| Metric | Normal | Pneumonia | COVID-19 | Overall |
|--------|--------|-----------|----------|---------|
| **Precision** | High | Good | High | Strong |
| **Recall** | High | Moderate | High | Reliable |
| **F1-Score** | High | Good | High | Balanced |
| **Specificity** | Excellent | Excellent | Excellent | Robust |

### Key Performance Indicators

- **High accuracy** in distinguishing Normal and COVID-19 cases
- **Strong specificity** across all classes
- **Balanced performance** with comprehensive metrics including:
  - Sensitivity (True Positive Rate)
  - Specificity (True Negative Rate)
  - Precision (Positive Predictive Value)
  - F1-Score (Harmonic mean of Precision and Recall)
  - Matthews Correlation Coefficient (MCC)
  - False Discovery Rate (FDR)
  - False Negative Rate (FNR)
  - Negative Predictive Value (NPV)

### Confusion Matrix Analysis

The model shows:
- **Excellent performance** on Normal and COVID-19 classification
- **Slightly higher misclassifications** between Pneumonia and COVID-19 (due to visual similarities)
- **Overall strong diagnostic capability** suitable for clinical support

## Technology Stack

### Backend
- **Language**: Python 3.7+
- **Framework**: Flask (Web application)
- **Deep Learning**: TensorFlow, Keras
- **Image Processing**: OpenCV (cv2)
- **Data Handling**: NumPy, Pandas
- **Model Evaluation**: scikit-learn

### Frontend
- **HTML5**: Structure and layout
- **CSS3 & Tailwind CSS**: Modern, responsive styling
- **JavaScript**: Interactive elements and dynamic content

### Data Visualization
- **Matplotlib**: Training/validation curves
- **Seaborn**: Statistical visualizations

### Dataset
- **Source**: COVID-19 Radiography Database (Kaggle)
- **Retrieved using**: kagglehub
- **Classes**: COVID-19, Normal, Viral Pneumonia

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- Git (optional, for cloning)

### Setup Instructions

1. **Clone or download the repository**
   ```bash
   git clone <repository-url>
   cd ML
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install flask tensorflow keras opencv-python numpy pandas matplotlib seaborn scikit-learn werkzeug
   ```

   Or use a requirements.txt if available:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify model files**
   - Ensure `CNN_Covid19_Xray_Version.h5` exists in the project directory
   - Ensure `Label_encoder.pkl` exists in the project directory

5. **Update file paths in app.py** (if necessary)
   ```python
   # Line 11-12 in app.py
   model = load_model('CNN_Covid19_Xray_Version.h5')
   le = pickle.load(open("Label_encoder.pkl", 'rb'))
   ```

6. **Create uploads directory** (automatically created by app)
   ```bash
   mkdir uploads
   ```

## Usage

### Running the Application

1. **Start the Flask development server**
   ```bash
   python app.py
   ```

2. **Access the web interface**
   - Open your web browser
   - Navigate to: `http://127.0.0.1:5000/` or `http://localhost:5000/`

3. **Application will start in debug mode** (for development)

### Using the Application

#### Method 1: Upload X-ray Image
1. Go to the home page (`/` or `/index`)
2. Click the file upload button
3. Select a chest X-ray image (PNG, JPG, JPEG formats)
4. Click submit/upload
5. View results page with:
   - Uploaded image display
   - Predicted class (Normal, Pneumonia, or COVID-19)
   - Confidence score (probability)

#### Method 2: Webcam Capture
1. Navigate to `/camera` endpoint
2. Grant camera permissions when prompted
3. Position X-ray image in front of camera
4. Capture image
5. View classification results

## System Workflow

```
User Upload → Image Preprocessing → CNN Model → Prediction → Result Display
     ↓              ↓                    ↓            ↓           ↓
  X-ray Image   Resize (150x150)    Feature      Softmax    Web Interface
                Normalize [0,1]    Extraction   Classifier   with Confidence
```

### Detailed Process Flow

1. **Input**: User uploads chest X-ray via web interface
2. **Secure Handling**: Filename sanitized using `secure_filename()`
3. **Preprocessing**: 
   - Image loaded using OpenCV
   - Converted to RGB format
   - Resized to 150x150 pixels
   - Normalized to [0, 1] range
   - Expanded dimensions for model input
4. **Prediction**: CNN model processes the image
5. **Output**: 
   - Predicted class decoded using Label Encoder
   - Confidence score extracted
   - Results rendered on result page

## Troubleshooting

### Common Issues and Solutions

#### 1. Model File Not Found
```python
# Error: FileNotFoundError: CNN_Covid19_Xray_Version.h5
# Solution: Verify model path
model = load_model('CNN_Covid19_Xray_Version.h5')
# Ensure the .h5 file is in the same directory as app.py
```

#### 2. Label Encoder Loading Error
```python
# Error: FileNotFoundError: Label_encoder.pkl
# Solution: Check pickle file location
le = pickle.load(open("Label_encoder.pkl", 'rb'))
```

#### 3. OpenCV Image Loading Fails
```python
# Error: Image returns None
# Solution: Check image path and format
# Supported formats: PNG, JPG, JPEG
# Ensure image is not corrupted
```

#### 4. TensorFlow/Keras Version Mismatch
```bash
# Error: Model loading fails due to version incompatibility
# Solution: Reinstall compatible versions
pip uninstall tensorflow keras
pip install tensorflow==2.x keras==2.x
```

#### 5. Uploads Folder Permission Error
```bash
# Error: Permission denied when saving files
# Solution: Ensure proper folder permissions
chmod 755 uploads/  # On Linux/macOS
# Or check folder exists and has write permissions on Windows
```

#### 6. Port Already in Use
```bash
# Error: Address already in use (Port 5000)
# Solution: Change port in app.py
app.run(debug=True, port=5001)
```

## Literature Review & References

This project builds upon extensive research in medical image analysis and deep learning:

### Key Research Papers

1. **CheXNet** (Rajpurkar et al., 2017) - Landmark 121-layer DenseNet for pneumonia detection
2. **COVID-Net** (Wang et al., 2020) - Custom CNN for COVID-19 detection using COVIDx dataset
3. **VGG19 Transfer Learning** (Apostolopoulos & Mpesiana, 2020) - Multi-class chest X-ray classification

### References

1. **Cohen, J. P., Morrison, P., & Dao, L.** (2020). *COVID-19 Image Data Collection*. arXiv:2003.11597. https://arxiv.org/abs/2003.11597

2. **Wang, L., Lin, Z. Q., & Wong, A.** (2020). *COVID-Net: A Tailored Deep Convolutional Neural Network Design for Detection of COVID-19 Cases from Chest X-Ray Images*. arXiv:2003.09871. https://arxiv.org/abs/2003.09871

3. **Chowdhury, M. E. H., et al.** (2020). *Can AI Help in Screening Viral and COVID-19 Pneumonia?* IEEE Access, 8, 132665-132676. DOI: 10.1109/ACCESS.2020.3010287

4. **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning*. MIT Press. https://www.deeplearningbook.org

5. **Keras Documentation** - https://keras.io

6. **COVID-19 Radiography Database**, Kaggle - https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

---
