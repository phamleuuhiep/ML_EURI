import tensorflow as tf
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

reddit_data = pd.read_csv('Reddit_Data.csv')
twitter_data = pd.read_csv('Twitter_Data.csv')
data = pd.concat([twitter_data, reddit_data], ignore_index = True)

X = data['messages1']
y = data['labels']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transform the testing data
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# If GPU is available, set it as the device
if tf.config.list_physical_devices('GPU'):
    try:
        with tf.device('/GPU:0'):
            # Initialize and train the SVM model
            svm_model = LinearSVC()
            svm_model.fit(X_train_tfidf, y_train)

            # Make predictions on the test set
            svm_predictions = svm_model.predict(X_test_tfidf)

            # Evaluate the model
            svm_f1 = f1_score(y_test, svm_predictions, average='weighted')
            print("SVM F1 Score:", svm_f1)

            svm_accuracy = accuracy_score(y_test, svm_predictions)
            print("SVM Accuracy Score:", svm_accuracy)

            print(classification_report(y_test, svm_predictions))
    except RuntimeError as e:
        print("GPU runtime error:", e)
        print("Falling back to CPU training...")
        svm_model = LinearSVC()
        svm_model.fit(X_train_tfidf, y_train)
        svm_predictions = svm_model.predict(X_test_tfidf)
        svm_f1 = f1_score(y_test, svm_predictions, average='weighted')
        print("SVM F1 Score:", svm_f1)
        svm_accuracy = accuracy_score(y_test, svm_predictions)
        print("SVM Accuracy Score:", svm_accuracy)
        print(classification_report(y_test, svm_predictions))
else:
    print("No GPU found, training on CPU")
    svm_model = LinearSVC()
    svm_model.fit(X_train_tfidf, y_train)

    # Make predictions on the test set
    svm_predictions = svm_model.predict(X_test_tfidf)

    # Evaluate the model
    svm_f1 = f1_score(y_test, svm_predictions, average='weighted')
    print("SVM F1 Score:", svm_f1)

    svm_accuracy = accuracy_score(y_test, svm_predictions)
    print("SVM Accuracy Score:", svm_accuracy)

    print(classification_report(y_test, svm_predictions))


# Initialize MLPClassifier model
mlp_model = MLPClassifier()

# Train the MLPClassifier model on the training data
mlp_model.fit(X_train_tfidf, y_train)
y_pred_mlp = mlp_model.predict(X_test_tfidf)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print("MLPClassifier Accuracy:", accuracy_mlp)
