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

Detail

------------------------------------------------------------
Input Layer                   Accepts grayscale or RGB images (depends on dataset preprocessing). 
                              Images are resized to a fixed dimension before being fed into the network.
Conv Layer 1                  Applies multiple convolutional filters (e.g., 3×3 or 5×5) to detect low-level 
                              features like edges and textures. Followed by batch normalization and activation (ReLU).
BatchNorm 1                   Normalizes activations to stabilize training and improve convergence.
ReLU Activation 1             Introduces non-linearity to learn complex patterns.
Max Pooling 1                 Downsamples feature maps to reduce computational cost while retaining critical information. 
                              (e.g., 2×2 pooling)
Conv Layer 2                  Extracts higher-level spatial features (e.g., shapes, patterns). 
                              Number of filters typically increases at this stage.
BatchNorm 2                   Normalizes activations again for stable training.
ReLU Activation 2             Non-linearity applied to feature maps.
Max Pooling 2                 Further reduces spatial dimensions while keeping important features.
Conv Layer 3                  Extracts deeper, more abstract features relevant for classification.
BatchNorm 3                   Normalization layer for stability.
ReLU Activation 3             Activation function for deeper layers.
Global Average Pooling        Reduces each feature map to a single value, replacing fully connected layers 
                              in some implementations. Helps reduce overfitting.
Flatten Layer                 Converts feature maps into a 1D vector to be processed by dense layers.
Fully Connected Layer 1 (FC1) First dense layer with a large number of neurons (e.g., 256, 512), 
                              learns complex feature representations.
Dropout Layer (if applied)    Randomly deactivates neurons to prevent overfitting.
Fully Connected Layer 2 (FC2) Reduces dimensionality further for classification.
Output Layer (Softmax)        Generates final class probabilities for classification 
                              (3 categories: Normal, Pneumonia-Viral, Pneumonia-Bacterial).


Training configuration
Loss function: The implementation includes Cross-Entropy Loss for multi-class classification.
Optimizer: Adam with starting learning rate 10e-4 for efficient weight updates.
Learning rate Scheduler: Learning rate is adjusted dynamically to improve convergence.

### Naive Bayes

### Genetic Algorithms
Phase 1 : using GA for optimizing CNN parameters, used for training CV dataset
