# 📡 Mahalanobis Distance Based Open-Set Recognition in RFFI Settings using Learnable Vector-Quantization
This repository presents a **Mahalanobis distance-based open set recognition framework** for radio frequency fingerprinting identification (**RFFI**). Leveraging **learnable vector quantization**, the model provides robust device classification and reliably detects unknown devices in real-world wireless environments.

**For further details, extended derivations, and more experimental results, see the full paper: [`ModelBasedOpenSetRecognition.pdf`](./ModelBasedOpenSetRecognition.pdf).**

---

## 📖 Overview

### 🔍 Motivation
Traditional RFFI approaches often assume a **closed-set** scenario, where all test devices are seen during training. However, in real world settings, new, previously unseen devices may appear a challenge known as **open set recognition (OSR)**.  
This work tackles open set recognition for RFFI, proposing a solution based on learnable vector quantization and Mahalanobis distance for improved generalization and unknown detection.

### 💡 Our Contribution
We propose the following innovations:
1. **Mahalanobis Distance-based Classification**:  
   Utilizes Mahalanobis distance in latent space, leveraging the full covariance structure for robust device classification and outlier detection.
2. **Open Set Evaluation**:  
   Provides comprehensive benchmarks for both seen (closed set) and unseen (open set) devices, highlighting the advantages of our approach.

---
## 🧮 Problem Formulation

We model latent features as class-specific clusters in high-dimensional space, assuming each device class follows a multivariate Gaussian distribution. Mahalanobis distance is used both for classifying known devices and detecting unknowns (open set recognition), with a threshold applied to the minimum class-wise distance for novelty detection.

---

## 🖥️ System Architecture

The pipeline below summarizes our method:

![system-architecture](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME/assets/your-architecture-diagram.png)

**Pipeline Steps:**
1. **Signal Acquisition**  
   - Raw RF signals are captured from wireless devices.
2. **Feature Extraction (Encoder)**  
   - A neural network encoder maps the signal to a compact latent representation.
3. **Learnable Vector Quantization**  
   - The latent features are quantized using a learnable codebook.
4. **Mahalanobis Distance-Based Classification**  
   - Device identity is inferred by measuring Mahalanobis distances in the codebook space. Unknown (open set) devices are detected as outliers.
5. **Open Set Detection**  
   - Thresholding the Mahalanobis score enables distinguishing between known and unknown devices.

---

## ✨ Features

- **Open Set Device Identification**:  
  Detects both known and previously unseen devices in realistic wireless settings.
- **Learnable Quantization**:  
  End-to-end training of the quantization codebook and encoder.
- **Statistically Robust Classification**:  
  Uses Mahalanobis distance, capturing codebook distribution beyond naive Euclidean proximity.
- **Flexible & Modular**:  
  Easily extensible to other signal types, quantization strategies, or OOD settings.

---

## 🏗️ Baseline: Five-Stage Training Pipeline

### Stage Descriptions

1. **Encoder-Classifier Training:**  
   Train a CNN encoder and MLP classifier jointly using cross-entropy loss.

2. **Encoder Fine-Tuning (Triplet Loss):**  
   Fine-tune the encoder with triplet loss to improve feature separability (classifier frozen).

3. **Classifier Retraining:**  
   Retrain a new classifier on the frozen encoder with cross-entropy loss.

4. **Codebook Construction (LBG Algorithm):**  
   Generate a codebook from encoder outputs using the LBG algorithm.

5. **Joint VQ-VAE Training:**  
   Initialize and train the full VQ-VAE model with all components.

The five-stage training procedure visual diagram:

| Stage | Encoder | Classifier | Codebook | Description                                               |
|-------|:-------:|:----------:|:--------:|-----------------------------------------------------------|
| 1     |   🔥    |     🔥     |    -    | Train encoder and classifier with cross-entropy loss      |
| 2     |   🔥    |     ❄️     |    -    | Fine-tune encoder with triplet loss (classifier frozen)   |
| 3     |   ❄️    |     🔥     |    -    | Retrain classifier on frozen encoder                      |
| 4     |   ❄️    |     ❄️     |    📦    | Build codebook with LBG algorithm (features fixed)        |
| 5     |   🔥    |     🔥     |    🔥    | Joint VQ-VAE training (all components trainable)          |

**Legend:**  
🔥 = trainable ❄️ = frozen 📦 = built in this stage

---

## 🔧 Proposed Modifications

- **Mahalanobis Triplet Loss:**  
  Use the class mean as the positive in triplet loss, and compute distances using the Mahalanobis metric (with regularized covariance).
- **Mahalanobis-Based Quantizer:**  
  Assign codewords in the VQ-VAE based on Mahalanobis distance to each codeword, with covariance estimated from the codebook (regularized).

---

## 📊 Dataset

- **Source:** [RF Fingerprinting With Deep Learning: A Robustness Analysis](https://arxiv.org/pdf/2107.02867)
- **Setup:** Device IDs 0–29 for training (closed set), 30–44 as unseen (open set).
- **Each sample:** Contains IQ data and device labels.

> See the [original paper](https://arxiv.org/pdf/2107.02867) for download and preprocessing details.

---

## 📈 Main Results

**Close-Set Classification Accuracy**

| Test Case                     | Ours (Mahalanobis) | Baseline (Euclidean) |
|-------------------------------|:------------------:|:--------------------:|
| 30 Devices (Validation Set)   | 84.77%             | **85.17%**           |
| 10 Unseen Devices (Test Set)  | **98.6%**          | 97.90%               |

**Open-Set Recognition (AUROC)**

| Metric       | Ours (Mahalanobis) | Baseline (Euclidean) |
|--------------|:------------------:|:--------------------:|
| Entropy      | **1.000**          | 0.9945               |
| Highest Peak | **1.000**          | 0.9977               |
| Mean         | **0.9998**         | 0.9901               |

---

## 📊 Codebook Usage Visualizations

The following figures visualize how the codebook is used in our Mahalanobis distance-based VQ-VAE.  
For each sample, we plot the inverse Mahalanobis distance $$\frac{1}{D_M(x, q)}$$, where $$D_M$$ is Mahalanobis distance
 to all codewords. 
Green bars indicate codewords belonging to the same class as the input.  
These plots help assess how codewords are activated and whether the model distributes assignments as intended.

---

### Multiple-Codeword Assignment

This plot shows a **histogram of the inverse Mahalanobis distance** $$\frac{1}{D_M(x, q)}$$ between a sample’s latent vector and all codewords in the codebook.  
**Green bars** correspond to codewords associated with the correct class for the input sample.

- The green bars stand out significantly from the others, clearly indicating the correct codewords that the model will assign.
- All other (non-matching) codewords have almost identical, low heights—demonstrating that the model’s latent space is well-distributed and codeword selectivity is strong.
- This pattern suggests that the model is confident in its assignments while maintaining a balanced use of codebook entries.

<img width="800" height="400" alt="codebook_usage_multi" src="https://github.com/user-attachments/assets/15289632-e435-42f2-bec2-3e7ee587413d" />

---

### Rogue Sample Histogram

This plot shows the **inverse Mahalanobis distance histogram** for a *rogue* (unknown) device sample.

- Mahalanobis-based models produce consistent histogram shapes for rogue samples, regardless of device.
- The lack of prominent standout bars (unlike the known-class sample) indicates the model is not overconfident about any codeword, helping to robustly detect open-set/unknown devices.
- This consistency across rogue devices contributes directly to improved **open set recognition** performance.

<img width="800" height="400" alt="rogue_histogram_mahalanobis" src="https://github.com/user-attachments/assets/dca870d9-47f1-4125-912f-70ff5541c575" />

---


## 👥 Contributors

This project was developed by:

- **Tal Kozakov** ([talkoz@post.bgu.ac.il](mailto:talkoz@post.bgu.ac.il))
- **Omer Hazan** ([hazanom@post.bgu.ac.il](mailto:hazanom@post.bgu.ac.il))

---

## 📂 References

- **Key Paper**:  
  [Your Paper Title](https://arxiv.org/abs/your-paper-arxiv-id) *(Add when available)*
- **Mahalanobis OOD Detection**:  
  [A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks](https://arxiv.org/abs/1807.03888)

---

For questions or model weights, please contact:  
- [talkoz@post.bgu.ac.il](mailto:talkoz@post.bgu.ac.il)  
- [hazanom@post.bgu.ac.il](mailto:hazanom@post.bgu.ac.il)

Feel free to explore the repository and contribute! 🚀
