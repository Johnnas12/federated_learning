# Federated Learning with TensorFlow Federated (TFF)

This project implements a Federated Learning (FL) system using the **EMNIST dataset** and **TensorFlow Federated (TFF)**. It simulates a federated environment where multiple clients train a shared model collaboratively while keeping their data decentralized.

---

## 📚 Project Overview

- **Framework**: TensorFlow Federated (TFF)
- **Dataset**: Federated EMNIST Digits
- **Objective**: Train a shared digit classification model using Federated Averaging without centralizing client data.
- **Approach**:
  - Load and preprocess federated EMNIST data
  - Define a model function for TFF
  - Configure a `tff.learning.algorithms.build_weighted_fed_avg`
  - Simulate multiple rounds of federated training
  - Track and plot metrics such as accuracy and loss

---

## 📦 Requirements

- Python 3.9+
- TensorFlow 2.14.1
- TensorFlow Federated 0.87.0
- NumPy
- Matplotlib

Install dependencies:
```bash
pip install tensorflow==2.14.1 tensorflow-federated==0.87.0 numpy matplotlib
