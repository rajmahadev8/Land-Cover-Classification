# 🌍 Land Cover Classification using ANN and CNN on EuroSAT Satellite Imagery

## Overview

This project presents a comparative deep learning study between **Artificial Neural Networks (ANN/MLP)** and **Convolutional Neural Networks (CNNs)** for **land cover classification** using the **EuroSAT satellite imagery dataset**.

The objective of this work is to understand how different neural network architectures learn from satellite imagery and experimentally demonstrate why CNNs outperform traditional fully connected neural networks in computer vision tasks.

The project includes:

- Exploratory Data Analysis (EDA)
- Satellite image preprocessing
- ANN/MLP model development
- CNN model development
- Hyperparameter tuning and optimization
- Regularization experiments
- GPU-accelerated training
- Comparative performance evaluation
- Confusion matrix and error analysis

---

# 🛰️ Dataset

The project uses the **EuroSAT dataset**, a benchmark dataset derived from Sentinel-2 satellite imagery from the European Space Agency’s Copernicus program.

### Dataset Characteristics

- ~27,000 RGB satellite image patches
- Resolution: **64 × 64 pixels**
- 10 land cover classes:
  - Annual Crop
  - Forest
  - Herbaceous Vegetation
  - Highway
  - Industrial
  - Pasture
  - Permanent Crop
  - Residential
  - River
  - Sea/Lake

The dataset provides diverse satellite imagery containing vegetation, urban structures, transportation infrastructure, and water bodies.

---

# 🎯 Objectives

The primary goals of this project were:

- Analyze statistical properties of satellite imagery through EDA
- Build and evaluate ANN/MLP architectures
- Build and evaluate CNN architectures
- Compare ANN and CNN performance for image classification
- Perform hyperparameter tuning for model optimization
- Investigate the impact of regularization methods
- Understand why CNNs outperform MLPs in spatial learning tasks

---

# 📊 Exploratory Data Analysis

Before model development, exploratory data analysis was conducted to better understand the EuroSAT dataset.

### EDA Included

- Class distribution visualization
- RGB channel intensity distributions
- Sample image visualization
- Spectral property analysis
- Spatial texture inspection

These analyses revealed distinct spectral and spatial characteristics across land cover classes.

---

# ⚙️ Data Preprocessing

The preprocessing pipeline included:

- JPEG image decoding
- Pixel normalization to `[0,1]`
- Stratified train-validation-test split (80-10-10)
- Dataset shuffling
- Mini-batch generation
- TensorFlow data prefetching

For ANN/MLP training, each image was flattened into:

```math
64 \times 64 \times 3 = 12288
```

input features per image.

---

# 🧠 ANN / MLP Architectures

Multiple MLP architectures were implemented and evaluated.

## Baseline MLP

- Single hidden layer
- 256 neurons
- Adam optimizer
- Polynomial learning rate decay

## Deep MLP

- Multiple hidden layers
- 512 neurons per layer
- L2 regularization
- Early stopping

## Hyperparameter-Tuned MLP

Grid search explored:

- Number of hidden layers
- Number of neurons
- Learning rate schedules
- L2 regularization strength
- Batch normalization
- Dropout

### Best MLP Performance

| Model | Train Accuracy | Validation Accuracy |
|---|---|---|
| Best MLP Model | 79% | 67% |

### Key Observation

MLP models were capable of learning global spectral patterns but struggled to capture spatial relationships within images.

---

# 🧩 CNN Architecture

For the CNN implementation, a modified **LeNet-5 architecture** was used.

The architecture consisted of:

- Convolutional layers for spatial feature extraction
- Max pooling layers for dimensionality reduction
- ReLU activation functions
- Fully connected dense layers
- Softmax output layer for multiclass classification

The CNN model was trained using:

- Adam optimizer
- Polynomial learning rate decay
- Early stopping
- TensorBoard monitoring
- GPU-accelerated training

Unlike MLPs, the CNN preserves spatial structure within satellite images and learns hierarchical image representations such as:

- textures
- edges
- shapes
- contextual patterns
- spatial relationships

This enables CNNs to significantly outperform traditional ANN architectures for image classification tasks.

---

# 🔬 CNN Hyperparameter Tuning

Extensive hyperparameter tuning was performed using GPU computing resources.

## Parameters Explored

| Hyperparameter | Values |
|---|---|
| Start Learning Rate | 1e-5, 1e-4, 1e-3 |
| L2 Regularization | 0, 1e-5, 1e-4 |
| Size Factor | 1, 2, 3 |

Grid search experiments were conducted to identify the best-performing CNN configuration.

---

# 🏆 Best CNN Results

## Best Hyperparameters

| Hyperparameter | Best Value |
|---|---|
| Start Learning Rate | 0.000100 |
| L2 Weight | 0.000000 |
| Size Factor | 2 |

## Final CNN Performance

| Metric | Value |
|---|---|
| Train Loss | 0.292 |
| Validation Loss | 0.454 |
| Train Accuracy | 90.20% |
| Validation Accuracy | 84.07% |

---

# 📈 Overall Results

| Model | Validation Accuracy |
|---|---|
| Baseline MLP | 58% |
| Deep MLP | 60% |
| Best MLP Model | 67% |
| Best CNN Model | 84% |

The CNN architecture achieved significantly higher performance than ANN/MLP models, demonstrating the importance of spatial feature extraction in computer vision tasks.

---

# ⚔️ ANN vs CNN Comparison

| Feature | ANN / MLP | CNN |
|---|---|---|
| Input Representation | Flattened Pixels | Spatial Feature Maps |
| Spatial Awareness | ❌ No | ✅ Yes |
| Texture Learning | Limited | Strong |
| Parameter Efficiency | Lower | Better |
| Generalization | Moderate | Better |
| Validation Accuracy | 67% | 84% |

---

# 🔍 Key Insights

This project experimentally demonstrates that:

- MLP models can learn global spectral information from satellite imagery
- CNNs significantly outperform MLPs because they preserve spatial structure
- Spatial feature extraction is critical for remote sensing image classification
- CNNs are substantially more effective for texture and contextual learning

The performance gap between ANN and CNN models clearly highlights why convolutional architectures dominate modern computer vision applications.

---

# 🚀 GPU-Accelerated Training

To efficiently train deep learning models and perform hyperparameter tuning experiments, GPU cluster resources were utilized.

## Training Infrastructure

- GPU cluster-based model training
- Accelerated deep learning experimentation
- Faster hyperparameter optimization
- Efficient large-scale model evaluation

Using GPU resources significantly reduced training time and enabled experimentation with deeper architectures and multiple hyperparameter configurations.

---

# 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

# 📂 Repository Structure

```bash
├── notebooks/
│   ├── ANN_Experiments.ipynb
│   ├── CNN_Experiments.ipynb
│   ├── Hyperparameter_Tuning.ipynb
│   └── Final_Project.ipynb
├── report/
│   └── Final_Report.pdf
├── figures/
│   ├── accuracy_curves/
│   ├── confusion_matrices/
│   └── sample_predictions/
├── models/
├── README.md
└── requirements.txt
```

---

# 🔮 Future Improvements

Potential future extensions include:

- Transfer learning using pretrained CNNs
- Vision Transformers (ViTs)
- Multispectral Sentinel-2 band utilization
- Explainable AI for remote sensing
- Attention mechanisms
- Data augmentation strategies
- Advanced geospatial segmentation models

---

# 📌 Conclusion

This project demonstrates the effectiveness of deep learning methods for satellite image classification and highlights the importance of spatial learning in computer vision.

While MLP models provide a useful baseline for understanding spectral learning, CNNs significantly outperform ANN architectures by preserving spatial relationships and learning hierarchical image features.

The study provides both practical implementation experience and theoretical insight into why convolutional architectures are the standard approach for modern image classification tasks.

---

# 👨‍💻 Author

**Raj Mahadevwala**

Master’s Student – Data Science / AI  
Technical University of Braunschweig

---
