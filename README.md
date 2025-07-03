# Breast Cancer Diagnosis - SVM Classification Workflow

This project involved binary classification using the Breast Cancer Wisconsin (Diagnostic) Dataset. The objective was to build, evaluate, and visualize SVM-based classification models.

## 1. Dataset Loading and Preparation

We started by loading the dataset and performing initial cleaning:
- Removed unnecessary columns (`id` and `Unnamed: 32`).
- Encoded the target column `diagnosis`, where malignant tumors were labeled as 1 and benign tumors as 0.
- This resulted in a clean dataset with 30 numerical features and 1 binary target variable.

## 2. Training SVM Models

We split the data into training and test sets using an 80-20 split and standardized all features using `StandardScaler` to ensure uniform feature scaling. Two SVM classifiers were trained:
- One with a linear kernel.
- Another with an RBF (Radial Basis Function) kernel.

We evaluated both models using accuracy score, confusion matrix, and a full classification report (precision, recall, F1-score).

## 3. Visualizing Decision Boundaries

To visualize how SVM models separate classes, we reduced the feature space to 2 dimensions using PCA. After dimensionality reduction:
- Linear and RBF SVMs were retrained on the PCA-reduced data.
- Decision boundaries were plotted using matplotlib, showing how each model separates the two diagnosis classes in 2D space.

## 4. Hyperparameter Tuning

To improve performance, we tuned the SVM hyperparameters using GridSearchCV:
- The grid search explored different values of `C` and `gamma` for the RBF kernel.
- A 5-fold cross-validation was used to evaluate each combination.
- The best performing parameters and cross-validation score were identified.
- The optimized model was then evaluated on the test set using classification metrics.

## 5. Cross-Validation Evaluation

We used 5-fold cross-validation to evaluate the final SVM model's generalization performance. Accuracy scores for each fold were reported, along with the mean and standard deviation. This step validated the consistency of model performance across different data splits.
