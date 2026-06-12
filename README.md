# 🚨 Surveillance Anomaly Detection — Model Comparison Study

This project compares different deep learning models for detecting traffic anomalies in surveillance camera footage. By extracting compact frame features and analyzing temporal sequences, the models learn to identify unusual events (e.g., reckless driving, illegal turns, wrong-way driving) without requiring explicit labels during training (unsupervised/self-supervised learning).

---

## 📋 Table of Contents
1. [Project Overview & Pipeline](#-project-overview--pipeline)
2. [Model Architectures](#-model-architectures)
3. [Model Comparison & Evaluation](#-model-comparison--evaluation)
4. [Streamlit Interactive Dashboard](#-streamlit-interactive-dashboard)
5. [Getting Started & Installation](#-getting-started--installation)

---

## 🔍 Project Overview & Pipeline

The system detects anomalies by evaluating how well a temporal autoencoder can reconstruct sequences of features. Since the model is trained exclusively on normal traffic behaviors, it fails to reconstruct anomalous sequences correctly, resulting in high reconstruction errors.

The complete detection pipeline is structured as follows:

```
Video Frames ──> ResNet50 (2048-D features) ──> Normalization ──> Sliding Window (w, s) ──> Autoencoder/VAE ──> Reconstruction MSE ──> Anomaly Score (MAX-scoring)
```

1. **Feature Extraction**: Video frames are passed through a pre-trained **ResNet50** CNN (classification head removed) to extract a **2048-dimensional feature vector** per frame, drastically reducing dimensionality.
2. **Normalization**: Features are standardized using training statistics only to avoid data leakage.
3. **Temporal Sequencing**: Frames are grouped into temporal sequences using a sliding window approach with parameters:
   - **Window Size ($w$)**: The number of consecutive frames in a sequence.
   - **Stride ($s$)**: The step size between sliding windows.
4. **Reconstruction & Scoring**: The model reconstructs the sequence. The frame-level Mean Squared Error (MSE) is computed. We apply a **MAX-scoring** method (selecting the maximum frame error within a window) to sensitize the system to short, sudden anomaly events.

---

## 🧠 Model Architectures

Three model configurations are compared in this study to evaluate the impact of window size and regularization on generalizability:

### 1. Transformer Autoencoder ($w=16$, $s=5$)
* **Description**: Utilizes a standard Transformer Encoder architecture with self-attention to capture temporal dependencies over a larger window of 16 frames (approx. 1.6 seconds).
* **Limitation**: Suffers from significant overfitting, yielding a high validation AUC but failing to generalize to the unseen test set.

### 2. Transformer Autoencoder ($w=8$, $s=5$)
* **Description**: Reduces the temporal window to 8 frames, narrowing the attention span.
* **Result**: Lower temporal complexity mitigates some overfitting, boosting test set performance.

### 3. Transformer Variational Autoencoder (VAE) ($w=8$, $s=5$)
* **Description**: Adds a probabilistic bottleneck (regularizing the latent space using Kullback-Leibler (KL) divergence). Instead of mapping inputs to fixed vectors, it encodes them as a distribution (mean $\mu$ and standard deviation $\sigma$).
* **Result**: The smooth, continuous latent space acts as a strong regularizer, preventing overfitting and leading to highly consistent validation and test performance.

---

## 📊 Model Comparison & Evaluation

All models were evaluated on the test dataset containing real annotated traffic anomalies. The target was to exceed **80% AUC-ROC** on the test set.

### Performance Summary

| Architecture | Window ($w$) | Stride ($s$) | Validation AUC-ROC | Test AUC-ROC | Key Evaluation Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Transformer AE** | 16 | 5 | 0.8168 | 0.4743 | High validation score but severe overfitting; fails on test set. |
| **Transformer AE** | 8 | 5 | **0.8524** | 0.6490 | Shorter window helps, but still overfits normal behavior. |
| **Transformer VAE** | 8 | 5 | 0.8095 | **0.8100** | **Best & Most Consistent. Probabilistic regularizer ensures excellent generalizability.** |

### Visualization of Results

Below are the visualization curves generated during the experiment:

* **ROC Curve Comparison**: Shows how the Transformer VAE dominates the test performance compared to standard Autoencoders.
  
  ![ROC Comparison](experiments/roc_comparison.png)

* **Training and Validation Loss**: Shows convergence behaviors and early stopping checks.
  
  ![Training Curves](experiments/training_curves.png)

---

## 🖥️ Streamlit Interactive Dashboard

An interactive UI dashboard is included in [`streamlit_app.py`](streamlit_app.py) to inspect the model's performance on the test dataset.

### Dashboard Features:
* **Interactive Timeline Scrubber**: View the anomaly score timeline plotted against the ground-truth anomaly intervals.
* **Sequence Inspector**: Scrub through sequence indexes using a slider to inspect actual video frames and observe the model's predictions.
* **Custom Threshold Configuration**: Interactively adjust the sensitivity threshold ($\sigma$) and scoring method (max, max_std, mean).
* **Top Anomaly Ranking**: Lists the highest-scoring sequences to quickly identify the most severe traffic incidents.

### Screenshots of the UI:

* **Timeline and Sequence Inspector View**:
  
  ![Streamlit Timeline Scrubber](ui/Screenshot%202026-03-06%20at%206.50.01%E2%80%AFAM.png)

* **Sequence Frame Progression & Event Indexing**:
  
  ![Streamlit Frame Inspector](ui/Screenshot%202026-03-06%20at%206.56.38%E2%80%AFAM.png)

---

## 🚀 Getting Started & Installation

### 1. Install Dependencies
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Extract Features
If you are running the training pipeline from scratch, open and run the Jupyter notebook:
```bash
jupyter notebook surveillance_anomaly_detection.ipynb
```
Features are saved into the `saved_features/` folder.

### 3. Run the Dashboard
To start the Streamlit web interface and visually inspect the results:
```bash
streamlit run streamlit_app.py
```
This opens the local server at `http://localhost:8501`.
