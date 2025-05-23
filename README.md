# 🔍 Generative Anomaly Detection using GANs and VAEs

This project explores **Generative AI-based anomaly detection** using **Generative Adversarial Networks (GANs)** and **Variational Autoencoders (VAEs)** across both image datasets and real-world applications. The pipeline involves training generative models to learn normal patterns and identify anomalies by reconstruction error or latent representation.

---

## 🎯 Objective

- Implement and compare generative deep learning models for detecting anomalies
- Evaluate GANs vs VAEs on image datasets
- Apply VAEs to real-world anomaly detection in finance, healthcare, and cybersecurity

---

## 📚 Datasets

1. **MNIST Digits** – Handwritten digits (0–9), grayscale 28x28 images  
2. **MNIST Fashion** – Zalando clothing images (e.g., shoes, shirts, bags)

Each dataset will be used to:
- Train generative models (GAN, VAE)
- Generate new samples
- Detect outliers based on learned data distribution

---

## 🧠 Models Implemented

### ✅ Generative Adversarial Network (GAN)
- Generator: Produces fake samples from random noise
- Discriminator: Distinguishes real vs. fake samples
- Trained with adversarial loss

### ✅ Variational Autoencoder (VAE)
- Encoder: Compresses data into latent distribution
- Reparameterization trick: Enables stochastic sampling
- Decoder: Reconstructs input from latent space
- Trained with reconstruction + KL divergence loss

---

## 🔬 Tasks Breakdown

### Part 1: 🧪 Exploratory Data Analysis (5%)
- Load MNIST & Fashion MNIST
- Preview sample images
- Analyze sample counts and class distribution

### Part 2: 🎨 Generative Adversarial Networks (25%)
- Train GAN on MNIST digits
- Train GAN on Fashion MNIST (e.g., shoe class)
- Generate:
  - 10 synthetic samples
  - 5 specific-digit samples (based on roll number)
  - 5 fashion-specific items

### Part 3: 🌀 Variational Autoencoders (25%)
- Train VAE on MNIST and Fashion datasets
- Visualize latent space using PCA/t-SNE
- Generate:
  - 10 generic samples
  - 5 specific-digit samples
  - Fashion-specific samples (e.g., shoes)

### Part 4: 📊 Comparison and Analysis (10%)
- Evaluate image quality, training stability, latent space
- Discuss model limitations and improvements

### Part 5: 🌍 Save the World with VAE (35%)
Use VAE to detect anomalies in a real-world financial or operational dataset:
- Finance: Fraudulent transactions
- Cybersecurity: Intrusion detection
- Healthcare: Misdiagnosis prediction
- Manufacturing: Predictive maintenance
- Energy: Grid failure prevention

---

## 📈 Evaluation Metrics

- GAN Loss: Binary Cross-Entropy
- VAE Loss: Reconstruction (BCE) + KL Divergence
- Qualitative evaluation via image quality
- Latent space visualization (PCA/t-SNE)

---

## 🛠 Tech Stack

- Python 3.9+
- PyTorch or TensorFlow/Keras
- NumPy, pandas, matplotlib, seaborn
- scikit-learn
- t-SNE / PCA
