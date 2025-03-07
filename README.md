# COMPREHENSIVE MACHINE LEARNING MODELS: IMPLEMENTATION AND COMPARISON (1st Stage) 

<div align='center'>

| **Student ID** | **Name** | **Task** | **Contribution** |
| -------------- | ---------------- | -------- | ---------------- |
| Write your id | Write your name | Write your task | Write your contribution |
| Write your id | Write your name | Write your task | Write your contribution |
| Write your id | Write your name | Write your task | Write your contribution |
| Write your id | Write your name | Write your task | Write your contribution |
| Write your id | Write your name | Write your task | Write your contribution |

</div>

## General
Machine learning has revolutionized the field of data science and rapidly evolved in many fields in recent years, providing powerful tools for extracting insights and knowledge from complex datasets. As data availability and computational power grow exponentially, machine learning models have become an essential component in various fields, from finance and healthcare to marketing and autonomous systems, offering diverse methods for tackling problems that were previously considered impossible or too time-consuming. By leveraging vast amounts of data, sophisticated algorithms, and computational resources, machine learning enables many systems to learn from patterns and make predictions with remarkable accuracy in an incredibly short amount of time, enhancing operational efficiency, state-of-the-art data-driven solutions. As a variety of models increase, however, selecting the most appropriate one for a given problem can be difficult. In this project, we explore four prominent machine learning models, Decision Tree, Neural Network, Naive Bayes, and Genetic Algorithm, through implementations, evaluations, and comparisons in terms of performance to determine the most effective model for a given task.

In general, each of these models brings unique strengths and capabilities, making them suitable for different types of problems and data characteristics. Decision trees are widely used for their simplicity and interpretability, providing clear decision-making processes that can be visualized and easily understood by humans. Neural networks, inspired by human brains, offer powerful learning capabilities for complex, non-linear relationships within data and are particularly effective in large-scale, unstructured data such as images. Naive Bayes, a probabilistic classifier, excels in handling high-dimensional datasets with strong assumptions of feature independence. Genetic algorithms, a type of evolutionary algorithm, offer an innovative approach to optimization problems by mimicking the process of natural selection to evolve solutions over generations.

The central objective of our project is to evaluate and compare the performance of these models on a common dataset to analyze how they handle classification tasks under various conditions and settings. By implementing these algorithms, we aim to provide a comprehensive understanding of their strengths, limitations, and suitability for a specific problem. We will also explore their behavior with respect to accuracy, computational efficiency, interpretability, and robustness.


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
