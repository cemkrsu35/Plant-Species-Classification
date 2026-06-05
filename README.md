# Plant Species Classification using Deep Learning

This repository contains the codebase and evaluation report for an AI-powered image recognition system designed to classify 226 distinct plant species. Developed as an Artificial Intelligence course project, this study explores and compares the efficacy of training a custom Convolutional Neural Network (CNN) from scratch versus utilizing advanced Transfer Learning architectures.

## Project Overview

Accurate plant identification is vital for agriculture, botany, and outdoor safety, but manual classification is often time-consuming and error-prone due to the vast number of similar species. This project solves this real-world problem by leveraging Deep Learning to automate classification across a massive, high-dimensional visual dataset.

## Dataset

The model was trained on a highly diverse, combined dataset comprising approximately 200,000 images. To address class imbalances, the dataset was carefully split into:
*   **Train Set:** 123,796 images (70%)
*   **Validation Set:** 26,528 images (15%)
*   **Test Set:** 26,528 images (15%)

*Sources combined include the Vegetable Image Dataset, Oxford 102 Flower Dataset, Ornamental Plant Images Dataset, Fruit-360, and Plant Type Datasets*[cite: 2].

## Models and Architecture

The study implemented and rigorously evaluated five distinct machine learning models using **PyTorch**:
1.  **Custom CNN:** A base model built from scratch featuring three convolution layers, max pooling, ReLU activations, and two fully connected layers.
2.  **VGG16 (Transfer Learning):** Utilizing pre-trained ImageNet weights with custom fully connected classification layers.
3.  **ResNet50 (Transfer Learning):** Leveraging residual blocks for deeper feature extraction.
4.  **MobileNetV3 (Transfer Learning):** Selected for its lightweight architecture and high computational efficiency.
5.  **EfficientNetB0 (Transfer Learning):** Selected for its compound scaling method balancing width, depth, and resolution.

## Optimization and Regularization

To ensure high generalization and prevent the models from memorizing training data (overfitting), the following techniques were applied:
*   **Data Augmentation:** Random affix, horizontal/vertical flips, Color Jitter, and Center Cropping.
*   **Regularization:** Dropout (up to 40%) and L2 Regularization to penalize large weights.
*   **Optimization:** Adam Optimizer with dynamic learning rate adjustments via `ReduceLROnPlateau` to escape local minima.
