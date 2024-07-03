# Cancer Prediction Project

This project uses various machine learning algorithms to predict cancer levels based on the given dataset. The primary focus is on exploring different classification techniques and evaluating their performance to determine the most effective model for predicting cancer levels.

This project aims to classify the severity level of cancer in patients using various machine learning algorithms. The dataset includes various features related to patient health and habits, such as smoking and chest pain, and the target variable is the cancer severity level. The project involves data preprocessing, exploratory data analysis, visualization, and the application of multiple machine learning algorithms to predict cancer severity levels.


### Machine Learning Models

1. **Random Forest Classifier**:
   - The data is split into training and testing sets.
   - The model is trained on the training set and predictions are made on the testing set.
   - Accuracy, log loss, and F1 score are calculated.

2. **K-Nearest Neighbors (KNN)**:
   - The optimal number of neighbors is determined using cross-validation.
   - The model is trained and evaluated.
   
3. **K-Means Clustering**:
   - K-Means clustering is applied and the clusters are mapped to the original labels.
   - The model is evaluated using accuracy, log loss, and F1 score.
   
4. **Decision Tree Classifier**:
   - A decision tree classifier is trained and evaluated.

5. **Support Vector Machine (SVM)**:
   - An SVM model is trained and evaluated.

Certainly! Here is the accuracy report formatted for a GitHub README:


## Accuracy Report

| Algorithm               | Log Loss Score | F1 Score   |
|-------------------------|----------------|------------|
| RandomForestClassifier  | 9.99           | 1.0        |
| KNeighborsClassifier    | 9.99           | 1.0        |
| KMeans                  | 15.06          | 0.14       |
| SVM                     | 0.69           | 0.98       |
| DecisionTreeClassifier  | 9.99           | 1.0        |


## Requirements

1. **Install the required libraries**:
    ```sh
    pip install -r requirements.txt
    ```

## To Run

1. **Run the notebook**:
    Open and run `cancer_pred.ipynb` in Jupyter Notebook to see the full workflow, from data preprocessing to model evaluation and visualization.

