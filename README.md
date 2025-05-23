# COMPREHENSIVE MACHINE LEARNING MODELS: IMPLEMENTATION AND COMPARISON (1st Stage) 

<div align='center'>

| **Student ID** | **Name** | **Task** | **Contribution** |
| -------------- | ---------------------- | --------------- | ----------------------- |
| 2252733        | Nguyen Duong Khanh Tam | Naive Bayes     |           20%           |
| 2252223        | Pham Le Huu Hiep       | Neural Network  |           20%           |
| 2252235        | Nguyen Viet Hoang      | Write README    |           20%           |
| 2460001        | Quentin Tripard        | Genetic Algorithm | 20% |
| 2152503        | Nguyen Ho Tien Dat     | Decision Tree   |           20%           |

</div>

## General
Machine learning has been rapidly evolving in recent years, with a dramatic increase in a variety of models in various fields, ranging from finance and healthcare to customer and autonomous systems, even into the field of biology. As the number of models continues to grow and evolve, however, selecting the most appropriate model for a given task can be difficult. In this project, we explore four prominent machine learning models, Decision Tree, Neural Network, Naive Bayes, and Genetic Algorithm, through implementations, evaluations, and comparisons in terms of performance to determine the most effective model for a given problem. The central objective of our project is to evaluate and compare the performance of these models on a common dataset to analyze how they handle classification tasks under various conditions and settings, then provide a comprehensive understanding of their strengths, limitations, and suitability for a specific task. We will also explore their behavior with respect to accuracy, computational efficiency, interpretability, and robustness.

### Decision trees
Throughout this process, we explore Decision Trees for image classification. Starting with a fully grown tree, we observe the ability of overfitting issues, leading us to apply K-Fold Cross-Validation and Grid Search to tune key hyperparameters like max depth and minimum samples for splits and leaves. To further refine the model, we perform post-pruning using cost complexity pruning (ccp_alpha) with the goal to achieve a better balance between accuracy and complexity, improving generalization.

### Neural networks
The architecture consists of convolutional layers that extract spatial features from input images using learnable filters, followed by ReLU activation functions to introduce non-linearity for better feature learning. Max pooling layers downsample feature maps to reduce computational cost while retaining essential information. Fully connected layers flatten the extracted features and pass them through dense layers for final classification, with a softmax layer converting raw predictions into class probabilities for multi-class classification.

### Naive Bayes
We explore the Naive Bayes classifier, a probabilistic model based on Bayes' theorem with an independence assumption. Through the process, we have covered theoretical foundations, practical implementation using scikit-learn, and application to datasets like Iris for classification tasks. Key steps include data preprocessing, model training, prediction, and performance evaluation using metrics like accuracy and F1-score. Visualizations help interpret results, providing a comprehensive understanding of Naive Bayes in real-world scenarios.

### Genetic Algorithms
Hyperparameter optimization plays a pivotal role in enhancing the performance of deep learning models, particularly Convolutional Neural Networks (CNN). Traditional methods like grid search or random search, can be computationally expensive and may not efficiently explore the vast hyperparameter space. To address this challenge, evolutionary solution like Genetic Algorithms (GA) offer an adaptive approach to discovering optimal configurations. In term of simulating natural selection, GA iteratively refine hyperparameters to improve model accuracy. 


# COMPREHENSIVE MACHINE LEARNING MODELS: IMPLEMENTATION AND COMPARISON (2nd Stage) 

<div align='center'>

| **Student ID** | **Name** | **Task** | **Contribution** |
| -------------- | ---------------------- | --------------- | ----------------------- |
| 2252733        | Nguyen Duong Khanh Tam | Ensemble Methods     |           20%           |
| 2252223        | Pham Le Huu Hiep       | Supported Vector Machine, PCA, RNN  |           20%           |
| 2252235        | Nguyen Viet Hoang      | Write README    |           20%           |
| 2460001        | Quentin Tripard        | Genetic Algorithm | 20% |
| 2152503        | Nguyen Ho Tien Dat     | HMM, Bayes Network   |           20%           |

</div>

### Support Vector Machine (SVM)
A supervised learning algorithm used for classification and regression. It finds the best boundary (hyperplane) that separates classes by maximizing the margin between data points of different classes.

### Principal Component Analysis (PCA)
An unsupervised dimensionality reduction technique that transforms data into a new coordinate system, keeping the most important features (principal components) that capture the most variance in the data.

### Recurrent Neural Network (RNN)
A type of neural network designed for sequential data. It has memory of previous inputs, making it suitable for tasks like time series, language modeling, and speech recognition.

### Bayesian Network
A probabilistic graphical model used for classification and inference. It represents conditional dependencies among variables using a directed acyclic graph, enabling reasoning under uncertainty by computing posterior probabilities based on observed features.

### Hidden Markov Model (HMM)
A generative sequence model used for temporal or sequential data. It models systems with hidden (latent) states and observed outputs, capturing the probabilities of state transitions and emissions to infer the most likely sequence of hidden states from observed data.

### Ensemble Methods  
A set of machine learning techniques that combine multiple models to improve accuracy, reduce bias, and enhance generalization, leveraging methods like bagging, boosting, and stacking to make more reliable predictions.

### Link github: https://github.com/phamleuuhiep/ML_EURI
