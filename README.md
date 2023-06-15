## credit-card-fraud-detection
This project focuses on detecting fraudulent credit card transactions using machine learning techniques. 
The goal is to develop a model that can effectively identify fraudulent transactions and distinguish them from legitimate ones.
## Introduction
Credit card fraud is a significant concern for financial institutions and cardholders worldwide. Detecting fraudulent transactions is essential to prevent financial losses and protect customers. This project utilizes machine learning algorithms to build a fraud detection system that can accurately identify fraudulent credit card transactions.

By analyzing patterns and characteristics of fraudulent transactions, the model can learn to differentiate them from legitimate transactions, enabling timely detection and prevention of fraudulent activities.
## Dataset
The dataset used in this project contains a large number of credit card transactions, including both fraudulent and legitimate transactions. This dataset is collected from Kaggle. This is the link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
 The dataset is highly imbalanced, with a small percentage of fraudulent transactions compared to legitimate ones. This class imbalance poses a challenge in training accurate models, requiring careful handling during preprocessing and model training.
 ## Technologies Used
The following technologies were used in this project:

Python: Programming language used for data preprocessing, model training, and evaluation.
scikit-learn: Machine learning library utilized for building and evaluating fraud detection models.
pandas: Library for data manipulation and analysis.
Matplotlib and Seaborn: used for data visualization
Oversampling to balance the dataset
Jupyter Notebook: Interactive environment employed for prototyping and development.
## Model Training
This project utilizes machine learning algorithms specially Logistic Regression to train fraud detection models.
Due to the imbalanced nature of the dataset, oversampling technique is employed to address the class imbalance problem. 
Various classification algorithms, such as decison Tree, Random Forests, and Gradient Boosting, could be also used to train the preprocessed data.
## Evaluation
The performance of the trained fraud detection models is evaluated using appropriate evaluation metrics, such as accuracy, precision, recall, and F1 score.
