# COMPREHENSIVE MACHINE LEARNING MODELS: IMPLEMENTATION AND COMPARISON (1st Stage) 

## General

### Decision trees

### Neural networks
Phase 1 using CNN model for training CV dataset
Model description
The implemented model is a Convolutional Neural Network (CNN) designed for CV-based image classification. It is built using PyTorch and consists of multiple convolutional layers, followed by activation functions, pooling layers, and fully connected layers for final classification.

Architecture
Convolutional layers: The main purposes of them is to extract spatial features from input images using learnable filters.
Activation function (ReLU): It introduces non-linearity for better feature learning.
Pooling layers (Max Pooling): They downsample feature maps to reduce computational cost while retaining key information.
Fully connected layers (FC Layers): They are the main part to flatten extracted features and pass them through dense layers for final classification.
Softmax layer : It converts raw predictions into class probabilities (for multi-class classification).


Training configuration
Loss function: The implementation includes Cross-Entropy Loss for multi-class classification.
Adam optimizer: Adam with starting learning rate 10e-4 for efficient weight updates.
Learning rate Scheduler: Learning rate is adjusted dynamically to improve convergence.

### Naive Bayes

### Genetic Algorithms
Phase 1 : Implementation of a genetic algorithm to optimize the hyperparameters of a Convolutional Neural Network (CNN). The goal is to find the best combination of hyperparameters (e.g., number of filters, dense units, learning rate, batch size) that maximizes the validation accuracy of the CNN.
