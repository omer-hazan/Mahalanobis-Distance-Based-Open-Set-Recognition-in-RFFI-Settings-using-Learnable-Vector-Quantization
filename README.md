# 📡 Mahalanobis Distance Based Open-Set Recognition in RFFI Settings using Learnable Vector-Quantization
This repository presents a **Mahalanobis distance-based open set recognition framework** for radio frequency fingerprinting identification (**RFFI**). Leveraging **learnable vector quantization**, the model provides robust device classification and reliably detects unknown devices in real-world wireless environments.

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

## 📊 Dataset

We evaluate our method using real-world and synthetic RFFI datasets, including devices held out for **open set** testing.  
- **Dataset Example**:  
  - Device IDs: 0–29 (train/closed set), 30–44 (open set/unseen).
  - Each sample consists of IQ sequences and device labels.

> *Dataset details and download instructions coming soon.*

---

## 📊 Codebook Usage Visualizations

The following figures visualize how the codebook is used in our Mahalanobis distance-based VQ-VAE.  
For each sample, we plot the inverse Mahalanobis distance (\(1/D_M(x, q)\)) to all codewords. Green bars indicate codewords belonging to the same class as the input.  
These plots help assess how codewords are activated and whether the model distributes assignments as intended.

---

### Single-Codeword Collapse
![Single-codeword collapse](figures/codebook_usage_single.png)
*A failure case where a class is mapped to just one codeword, limiting expressiveness.*

---

### Collapse onto Class 3
![Collapse onto class 3](figures/codebook_usage_class3.png)
*Another failure case where the codebook collapses, over-assigning codewords to class 3 and leaving other classes underrepresented.*

---

### Multiple-Codeword Assignment
![Multiple-codeword assignment](figures/codebook_usage_multi.png)
*A more balanced scenario where multiple codewords are assigned per class.*

---

### Euclidean Quantization
![Euclidean quantization](figures/codebook_usage_euclidean.png)
*For comparison, the Euclidean case shows a more even codeword distribution—closer to the ideal number per class.*

---

### Rogue Sample Histogram
![Rogue sample histogram](figures/rogue_histogram_mahalanobis.png)
*Histogram for a rogue (unknown) device. Mahalanobis-based models yield more consistent rogue sample responses across devices, aiding open set recognition.*

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
