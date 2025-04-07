# Anomaly Detection in Household Electrical Appliances using Graph Neural Networks

## Overview

This project adapts the Graph Convolutional Network (GCN) and Transformer-based architecture for anomaly detection in household electrical appliances. The model uses reconstruction error to detect anomalies in the .npz files, which are the appliance data of a dishwasher. To evaluate the model, There are anomalous datasets as well. Evaluation metrics (accuracy, precision, recall, F1-score) are computed.

---

## Step-by-Step Implementation Guide

### 1. Dataset Preparation

#### **1.1 Load and Preprocess Data**

- All the data is stored in a folder called "dishwasher-dataset"
- All the data is stored in .npz files
- The folder contains healthy data and unhealthy data as well
- The model will learn from healthy data and test on the unhealthy data
- All the info about the dataset is in 100_Code_for_Visualizing_Healthy_Unhealthy_Activations.ipynb file.

---

### 2. Graph Construction

Follow the paper’s method to convert time series into graphs:

#### **2.1 Window Clustering (Nodes)**

- **Entropy Calculation**: Compute entropy for each appliance’s consumption window.
- **Clustering**: Use K-means (Lloyd’s algorithm) to cluster windows into `N` nodes based on entropy. Each node represents an operational state.
- **Node Embeddings**: Initialize node features as the mean of assigned windows (Eq. 3 in the paper).

#### **2.2 Transition Probabilities (Edges)**

- **Markov Chain**: Calculate transition probabilities between clusters using Eq. 4-5.
- **Adjacency Matrix**: Build a weighted directed graph where edges represent transition likelihoods.

---

### 3. Model Architecture

Adapt the paper’s seq2seq architecture for reconstruction:

#### **3.1 Encoder (GCN)**

- **Layers**: 8 GCN layers with ReLU activation (last layer: sigmoid).
- **Input**: Graph nodes (operational states) and adjacency matrix.
- **Output**: Latent graph embeddings capturing time-invariant dependencies.

#### **3.2 Decoder (Transformer)**

- **Layers**: 2 Transformer layers with multi-head attention (Eq. 8-9).
- **Output Layer**: 1D deconvolution + linear layers to reconstruct the input window.
- **Loss Function**: Mean Absolute Error (MAE) between input and reconstructed signal.

---

### 4. Training

- **Input**: Normal training data (no anomalies).
- **Optimizer**: Adam with learning rate = 0.001.
- **Batch Size**: 32.
- **Early Stopping**: Monitor validation loss (patience = 10 epochs).

---

### 5. Anomaly Detection

#### **5.1 Reconstruction Error**

- **Test Inference**: Reconstruct test windows using the trained model.
- **Error Metric**: Compute MAE/MSE for each window:
  \[
  \text{Error}_t = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}\_i|
  \]

#### **5.2 Threshold Determination**

- **Validation Set**: Use 20% of training data to calculate the 95th percentile of MAE as the threshold.
- **Anomaly Flag**: If \(\text{Error}\_t > \text{threshold}\), mark as anomalous.

---

### 6. Evaluation and Visualization

#### **6.1 Metrics**

- **Confusion Matrix**: Compare predicted vs. actual labels.
- **Scores**: Calculate accuracy, precision, recall, F1-score.

#### **6.2 Graphs**

1. **Reconstruction Error Over Time**:
   - Plot error values with threshold line; highlight anomalous regions.
2. **Normal vs. Anomalous Cycle**:
   - Overlay ground truth and reconstructed signals for a normal and anomalous window.
3. **Confusion Matrix**:
   - Heatmap showing TP, TN, FP, FN.
4. **Performance Bar Chart**:
   - Compare accuracy, precision, recall, F1-score.

---

### 7. Tools and Libraries

- **Python Libraries**:
  - PyTorch Geometric (GCN implementation).
  - PyTorch (Transformer, training pipeline).
  - Scikit-learn (clustering, metrics).
  - Pandas/NumPy (data processing).
  - Matplotlib/Seaborn (visualization).

---

### 8. Expected Challenges and Solutions

1. **Graph Construction**:
   - Challenge: Entropy calculation requires appliance-specific data.
   - Solution: Use aggregate signal entropy for simplicity.
2. **Threshold Tuning**:
   - Challenge: Fixed thresholds may not generalize.
   - Solution: Dynamic thresholds based on rolling window statistics.

---

## Conclusion

This project extends the GCN-Transformer architecture to detect anomalies in household energy data. By leveraging reconstruction error and synthetic anomalies, it provides a robust framework for identifying irregular appliance behavior. Future work includes testing on real anomalies and multi-appliance scenarios.
