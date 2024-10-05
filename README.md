## SCT_DS_3

## Bank Marketing Data Analysis and Decision Tree Classifier

This repository features a script written in Python that examines the "Bank Marketing" dataset sourced from the UCI Machine Learning Repository. The main objective is to forecast if a client will opt for a term deposit by utilizing various attributes through a Decision Tree Classifier. You can access the code in the project’s code file.

The analysis encompasses:-
->Retrieving and preparing data from the UCI Repository.
->Encoding categorical variables for use in machine learning models.
->Training a Decision Tree Classifier.
->Assessing the model's performance using accuracy and various other metrics.

The dataset utilized in this analysis is the "Bank Marketing" dataset, which comprises:
->Features: Client-related attributes such as age, occupation, marital status, education level, and additional characteristics.
->Target: Indicates whether the client subscribed to a term deposit (binary outcome).

Coding Details:-
1. Data Retrieval and Preparation
The script employs the ucimlrepo library to obtain the dataset. It separates the features from the target variable, with categorical features being encoded through one-hot encoding.

2. Model Training
The dataset is divided into training and testing subsets (80%-20%).
A Decision Tree Classifier is trained using the training subset.

3. Model Evaluation
The accuracy of the model is assessed on the testing subset.
A classification report and confusion matrix are produced to evaluate the model's effectiveness.

