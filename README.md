# COMPREHENSIVE MACHINE LEARNING MODELS: IMPLEMENTATION AND COMPARISON (1st Stage) 

<div align='center'>

| **Student ID** | **Name** | **Task** | **Contribution** |
| -------------- | ---------------------- | --------------- | ----------------------- |
| 2252733        | Nguyen Duong Khanh Tam | Naive Bayes     |           20%           |
| 2252223        | Pham Le Huu Hiep       | Neural Network  |           20%           |
| Write your id  | Write your name        | Write your task | Write your contribution |
| Write your id  | Write your name        | Write your task | Write your contribution |
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
### Link github: https://github.com/phamleuuhiep/ML_EURI
