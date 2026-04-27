# Hybrid Anomaly Detection with Representation Learning

This repository contains the implementation of the hybrid anomaly detection framework proposed in our study.

## Overview

The framework combines:

- Autoencoder-based representation learning  
- Multiple unsupervised anomaly detection algorithms  
- Pseudo-labeling based on reconstruction error  
- SHAP-based explainability  

The main objective is to evaluate how transforming the feature space into a latent representation affects anomaly detection performance.

---

## Dataset

The experiments are conducted on the **C-MAPSS (FD001)** dataset provided by NASA.

Dataset link:  
https://data.nasa.gov/

---

## Usage

The entire workflow is implemented in a single notebook:

```bash
Hybrid_Anomaly_Detection.ipynb
