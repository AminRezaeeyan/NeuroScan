# 🧠 NeuroScan - Brain Tumor Classifier

![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue?logo=kaggle&link=https://www.kaggle.com/code/aminrezaeeyan/brain-tumor-classification-using-cnn-98-32)
![Accuracy](https://img.shields.io/badge/accuracy-98.32%25-brightgreen)
![PyPI - Python Version](https://img.shields.io/badge/python-3.9%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18.1-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-3.0-red?logo=keras)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey?logo=flask)
![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)
![License](https://img.shields.io/badge/license-MIT-green)
![GitHub last commit](https://img.shields.io/github/last-commit/AminRezaeeyan/NeuroScan?color=blue&logo=github)

<details>
<summary><strong>📌 Table of Contents</strong></summary>

- [Model Overview](#-model-overview)
- [Key Features](#-key-features)
- [Web Interface](#-web-interface)
- [Installation](#-installation)
  - [Manual Setup](#manual-installation)
  - [Docker Setup](#docker-installation)
- [Usage](#-usage)
- [Performance Metrics](#-performance-metrics)
- [Future Work](#-future-work)
- [Contributing](#-contributions)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

</details>

---

## 🧩 Model Overview

A powerful CNN model that classifies brain tumors into four categories with **98.32% accuracy**:

1. **Glioma** - Irregular tumors originating from glial cells
2. **Meningioma** - Well-defined tumors of the meninges
3. **Pituitary** - Hormone-affecting glandular tumors
4. **No-tumor** - Healthy brain scans

<div align="center">
  <img src="docs/images/samples.png" width="100%">
</div>

---

## 🚀 Key Features

- **98.32% Test Accuracy** - State-of-the-art performance
- **VGG-Inspired Architecture** - Optimized 3x3 kernel design
- **Advanced Training**:
  - Early stopping & model checkpointing
  - Dynamic learning rate scheduling
  - Real-time data augmentation (rotations/flips)
- **Production-Ready**:
  - Flask web interface
  - Docker container support
  - REST API endpoints

<div align="center">
  <img src="docs/images/architecture1.svg" width="100%">
  <img src="docs/images/learning_curve.png" width="100%">
</div>

---

## 🌐 Web Interface

<div align="center" style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap; gap: 10px;">
  <img src="docs/images/webui1.png" style="height: 350px; width: auto; object-fit: contain; flex: 1 1 auto; max-width: 30%; min-width: 250px;">
  <img src="docs/images/webui2.png" style="height: 350px; width: auto; object-fit: contain; flex: 1 1 auto; max-width: 30%; min-width: 250px;">
  <img src="docs/images/webui3.png" style="height: 350px; width: auto; object-fit: contain; flex: 1 1 auto; max-width: 30%; min-width: 250px;">
</div>

**Features**:
- Drag-and-drop MRI upload
- Real-time visualization
- Detailed confidence reports
- Mobile-responsive design

---

## 📥 Installation

### Manual Installation
``` bash
# Clone repository
git clone https://github.com/AminRezaeeyan/NeuroScan.git
cd NeuroScan

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run development server
flask run --host=0.0.0.0 --port=5000
```
### Docker Installation
``` bash
# Clone repository
git clone https://github.com/AminRezaeeyan/NeuroScan.git
cd NeuroScan

# Build and run containers (detached mode)
docker-compose up -d --build

# View logs (optional)
docker-compose logs -f

```

---

## 💻 Usage

### Programmatic API

The pre-trained model (`best_model.keras`) is available in the `models/` directory and can be integrated into your applications:

```python
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('models/best_model.keras')

def predict_tumor(image_path):
    """
    Predicts tumor class from MRI image
    Args:
        image_path: Path to MRI image (JPEG/PNG)
    Returns:
        dict: {'class': 'glioma/meningioma/pituitary/notumor',
              'confidence': float,
              'probabilities': dict}
    """
    # Load and preprocess image using Pillow
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize to 224x224
    img = img.convert('RGB')  # Ensure RGB format
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    pred = model.predict(img_array)
    classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

    return {
        'class': classes[np.argmax(pred)],
        'confidence': float(np.max(pred)),
        'probabilities': {cls: float(prob) for cls, prob in zip(classes, pred[0])}
    }

# Example usage
result = predict_tumor("path/to/mri.jpg")
print(result)
```

Sample Output:
```
{
    'class': 'Meningioma',
    'confidence': 0.9743,
    'probabilities': {
        'Glioma': 0.0121,
        'Meningioma': 0.9743,
        'Pituitary': 0.0087,
        'No Tumor': 0.0049
    }
}
```

⚠️ **Important Note:**
The model automatically normalizes input images in its first layer. Do not manually normalize (divide by 255) as this will cause incorrect predictions. Simply pass the raw image array (0-255 values).

---

## 📊 Performance Metrics

| Metric        | Score  |
| ------------- | ------ |
| Accuracy      | 98.32% |
| Precision     | 98.32% |
| Recall        | 98.32% |
| F1 Score      | 98.32% |

<details>
<summary><b>Class-wise Breakdown</b></summary>

| Class          | Precision | Recall | F1 Score |
| -------------- | --------- | ------ | -------- |
| Glioma         | 98.99%    | 98.33% | 98.66%   |
| Meningioma     | 97.67%    | 96.08% | 96.87%   |
| Pituitary      | 98.66%    | 98.33% | 98.50%   |
| No-tumor       | 98.06%    | 100%   | 99.02%   |

</details>

---

## 🔮 Future Work

- **Continuous Accuracy Improvements**: Currently targeting 99%+ through:
  - Advanced attention mechanisms
  - Transformer-based hybrid architectures
  - Improved data augmentation pipelines

- **Tumor Segmentation**: Developing pixel-level detection

- **Clinical Integration**:
  - DICOM/PACS support
  - HL7/FHIR compatibility
  - Multi-modal analysis (MRI + CT)

- **Edge Deployment**:
  - ONNX runtime optimization

---

## 🤝 Contributions

### How to Contribute:
###
1. Fork the repository
2. Create your feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request
###

### Priority Areas:
- Improved model interpretability
- Additional medical imaging formats (DICOM, NIfTI)
- Performance optimizations
- UI/UX enhancements
- Documentation improvements

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
