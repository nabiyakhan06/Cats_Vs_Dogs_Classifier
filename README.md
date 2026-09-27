# 🐱 vs 🐶 Image Classification using CNN (Keras / TensorFlow)

A Convolutional Neural Network (CNN) built with TensorFlow and Keras to classify images into **Cats** and **Dogs**. The project demonstrates an end-to-end computer vision workflow, including data preprocessing, custom CNN architecture, training visualization, and single-image inference with OpenCV.

---

## 📌 Project Overview
- **Framework:** TensorFlow 2.x / Keras
- **Input Dimensions:** `(256, 256, 3)`
- **Architecture:** 3 Convolutional Blocks with Batch Normalization + Max Pooling
- **Classification Head:** Dense layers with Dropout regularization and Sigmoid activation
- **Task Type:** Binary Classification (`binary_crossentropy`)

---

## 🏗️ Model Architecture Diagram

```
Input Image (256 x 256 x 3)
         │
         ▼
┌─────────────────────────────────────────────────┐
│              Conv2D (32 Filters, 3x3)           │ ──► Shape: (254, 254, 32)
│              BatchNormalization()               │
│              MaxPooling2D (2x2, Stride=2)        │ ──► Shape: (127, 127, 32)
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│              Conv2D (64 Filters, 3x3)           │ ──► Shape: (125, 125, 64)
│              BatchNormalization()               │
│              MaxPooling2D (2x2, Stride=2)        │ ──► Shape: (62, 62, 64)
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│              Conv2D (128 Filters, 3x3)          │ ──► Shape: (60, 60, 128)
│              BatchNormalization()               │
│              MaxPooling2D (2x2, Stride=2)        │ ──► Shape: (29, 29, 128)
└─────────────────────────────────────────────────┘
         │
         ▼
   Flatten()                                      ──► Shape: (107648)
         │
         ▼
   Dense (128 units, ReLU) + Dropout(0.1)          ──► Shape: (128)
         │
         ▼
   Dense (64 units, ReLU) + Dropout(0.1)           ──► Shape: (64)
         │
         ▼
   Dense (1 unit, Sigmoid)                        ──► Output Probability: [0.0 - 1.0]
```

### Detailed Layer Specifications

| Layer Type | Configuration | Output Shape | Parameters |
| :--- | :--- | :--- | :--- |
| **Input** | RGB Image | `(None, 256, 256, 3)` | 0 |
| **Conv2D** | 32 Filters (3x3), ReLU | `(None, 254, 254, 32)` | 896 |
| **Batch Normalization** | Default | `(None, 254, 254, 32)` | 128 |
| **MaxPooling2D** | Pool Size (2x2), Stride 2 | `(None, 127, 127, 32)` | 0 |
| **Conv2D** | 64 Filters (3x3), ReLU | `(None, 125, 125, 64)` | 18,496 |
| **Batch Normalization** | Default | `(None, 125, 125, 64)` | 256 |
| **MaxPooling2D** | Pool Size (2x2), Stride 2 | `(None, 62, 62, 64)` | 0 |
| **Conv2D** | 128 Filters (3x3), ReLU | `(None, 60, 60, 128)` | 73,856 |
| **Batch Normalization** | Default | `(None, 60, 60, 128)` | 512 |
| **MaxPooling2D** | Pool Size (2x2), Stride 2 | `(None, 29, 29, 128)` | 0 |
| **Flatten** | Reshape to 1D | `(None, 107648)` | 0 |
| **Dense** | 128 Units, ReLU | `(None, 128)` | 13,779,072 |
| **Dropout** | Rate: 0.1 | `(None, 128)` | 0 |
| **Dense** | 64 Units, ReLU | `(None, 64)` | 8,256 |
| **Dropout** | Rate: 0.1 | `(None, 64)` | 0 |
| **Dense (Output)** | 1 Unit, Sigmoid | `(None, 1)` | 65 |

- **Total Parameters:** 13,881,141
- **Trainable Parameters:** 13,880,693
- **Non-trainable Parameters:** 448

---

## 📁 Dataset Directory Structure

Organize your training and validation data into subfolders named after the target classes. The dataset loader automatically infers class labels from folder names (`cats` $\rightarrow$ `0`, `dogs` $\rightarrow$ `1`).

```text
cat-vs-dog-cnn/
│
├── train/
│   ├── cats/    <-- Place training cat images here
│   └── dogs/    <-- Place training dog images here
│
├── test/
│   ├── cats/    <-- Place validation cat images here
│   └── dogs/    <-- Place validation dog images here
│
├── dog.jpg      <-- Sample image for inference
├── cat1.jpg     <-- Sample image for inference
└── main.ipynb   <-- Jupyter notebook containing project code
```

---

## ⚙️ Installation

### Option 1: Local Machine Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR-USERNAME/cat-vs-dog-cnn.git](https://github.com/YOUR-USERNAME/cat-vs-dog-cnn.git)
   cd cat-vs-dog-cnn
   ```

2. **Create and activate a virtual environment (Recommended):**
   - **Linux/macOS:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - **Windows:**
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```

3. **Install required dependencies:**
   ```bash
   pip install tensorflow matplotlib opencv-python notebook
   ```

---

### Option 2: Google Colab Setup

1. Open [Google Colab](https://colab.research.google.com/).
2. Select **File > Upload notebook** and upload `main.ipynb` (or import directly from your GitHub repository).
3. Ensure GPU acceleration is enabled:
   - Go to **Runtime > Change runtime type**.
   - Select **T4 GPU** (or available GPU accelerator) and click **Save**.
4. Upload your dataset zip file to `/content/` and extract it:
   ```python
   !unzip -q /content/dataset.zip -d /content/
   ```

---

## 🚀 Usage

### 1. Launch the Notebook
If running locally, start Jupyter Notebook:
```bash
jupyter notebook
```
Open `main.ipynb` in your browser.

---

### 2. Step-by-Step Execution Guide

- **Cell 1 — Import Libraries & Load Data:** Loads training and validation datasets from directory structures using `image_dataset_from_directory`.
- **Cell 2 — Preprocessing:** Maps input image tensors to a normalized pixel floating-point range $[0.0, 1.0]$ via `tf.cast(image / 255.0, tf.float32)`.
- **Cell 3 — Build Model:** Defines the sequential CNN layer architecture and prints `model.summary()`.
- **Cell 4 — Train Model:** Compiles with `adam` optimizer and `binary_crossentropy` loss, then fits over 10 training epochs.
- **Cell 5 — Visualize Performance:** Generates loss and accuracy plots comparing training vs. validation curves.
- **Cells 6 & 7 — Single Image Inference:** Run individual predictions on sample test images (`dog.jpg` and `cat1.jpg`).

---

## 🔬 Single Image Inference Pipeline

When passing new test images to `model.predict()`, the script executes these necessary preprocessing steps:
1. Load image using **OpenCV** (`cv2.imread`).
2. Convert color format from **BGR to RGB** (`cv2.cvtColor`).
3. Resize image tensor to **256x256** matching input shape.
4. Normalize pixel range by dividing by **255.0**.
5. Reshape to include batch dimension `(1, 256, 256, 3)`.