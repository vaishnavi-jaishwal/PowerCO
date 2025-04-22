# PowerCO
Customer Churn Prediction - Step 1: EDA, Preprocessing, and Modeling
This repository contains the initial step of a customer churn prediction project using Random Forest classification. The goal is to analyze customer data, preprocess it, engineer features, build a predictive model, and evaluate its performance to identify customers likely to churn.

Overview
This step involves:

Exploratory Data Analysis (EDA) to understand data distribution, missing values, and feature relationships.

Data preprocessing including merging additional pricing data, encoding categorical variables, handling missing values, and feature scaling.

Training a Random Forest classifier with hyperparameter tuning via Grid Search.

Model evaluation using metrics such as classification report, confusion matrix, ROC AUC score, and ROC curve visualization.

Analysis of feature importance to identify key predictors of churn.

Saving the trained model and scaler for future predictions.

Files and Data
data/client_data.csv — Main customer dataset with churn labels.

data/data_for_predictions.csv — Dataset containing customer features for model training.

data/price_data.csv — Price information merged with main features to enrich the dataset.

models/ — Directory where the trained model and scaler are saved.

Workflow Description
1. Data Loading and Inspection
Loads three CSV files: customer data, features for prediction, and price data.

Displays initial rows for quick inspection.

Visualizes the churn target distribution to check for class imbalance.

Checks for missing values and prints summary statistics.

2. Exploratory Data Analysis (EDA)
Visualizes the relationship between churn and key numeric features like consumption (cons_12m) and number of active products (nb_prod_act) using boxplots.

Plots a correlation heatmap to identify feature interrelationships.

3. Data Preprocessing
Fixes column inconsistencies in price data, aggregates it by customer ID, and merges it with the main prediction dataset.

Removes unnecessary columns like unnamed indices.

Encodes categorical features using Label Encoding.

Drops rows with missing values to ensure clean input.

Splits features (X) and target (y) variables.

Applies standard scaling to normalize feature distributions.

4. Modeling
Splits data into train and test sets with stratification to maintain class proportions.

Trains a Random Forest classifier with balanced class weights.

Performs hyperparameter tuning via Grid Search on:

Number of trees (n_estimators): 100, 200

Tree depth (max_depth): 10, 20, None

Minimum samples per split (min_samples_split): 2, 5, 10

Selects the best model based on ROC AUC score.

5. Evaluation
Predicts churn on test set and prints classification report with precision, recall, F1-score.

Shows confusion matrix to visualize true/false positives and negatives.

Calculates and displays ROC AUC score.

Plots ROC curve for model performance visualization.

6. Feature Importance
Displays the top 15 features contributing most to the model’s predictions to gain insight into churn drivers.

7. Model Persistence
Saves the trained Random Forest model and scaler object to the models/ directory for later use in deployment or prediction tasks.

How to Use
Clone the repo and install required packages:

bash
Copy
Edit
pip install pandas numpy matplotlib seaborn scikit-learn joblib
Place the CSV data files in the data/ folder as per the filenames used.

Run the script/notebook to execute the full workflow from data loading to model saving.

Next Steps
Perform more advanced feature engineering and handle class imbalance techniques.

Experiment with other models and ensemble methods.

Deploy the saved model for batch or real-time churn predictions.

Build dashboards to monitor churn risk and customer retention strategies.

If you have any questions or want to contribute, feel free to open an issue or pull request!

