# Customer Churn Analysis and Prediction

## Project Overview
This project aims to analyze customer churn behavior and prepare the dataset for machine learning modeling. By exploring customer demographics, service usage, and contract types, the project provides insights into patterns that lead to churn and sets the foundation for predictive modeling.

## Dataset
The dataset, `preprocessed_dataset.csv`, contains 7,043 customer records with the following 10 features:

- **gender**: Customer gender (Male/Female)  
- **SeniorCitizen**: Indicates if the customer is a senior citizen (0 = No, 1 = Yes)  
- **Dependents**: Whether the customer has dependents (Yes/No)  
- **tenure**: Number of months the customer has stayed with the company  
- **PhoneService**: Whether the customer has phone service (Yes/No)  
- **MultipleLines**: Whether the customer has multiple phone lines (Yes/No)  
- **InternetService**: Type of internet service (DSL/Fiber optic)  
- **Contract**: Contract type (Month-to-month/One year/Two year)  
- **MonthlyCharges**: Customer's monthly charges  
- **Churn**: Whether the customer churned (Yes/No)  

## Key Steps

### 1. Data Loading and Inspection
- Load dataset using `pandas.read_csv()`.  
- Preview first few rows with `churn.head()` and check dataset shape, column names, and info.  
- Perform initial exploratory analysis using `value_counts()` to understand distributions of categorical variables.  
- Check for missing values to ensure data completeness.

### 2. Data Preprocessing
- Convert categorical columns into numeric formats as needed (e.g., `Churn` → `Churn_binary`).  
- Handle missing values: numeric columns filled with median values, categorical columns filled with `"Missing"`.  
- Perform **one-hot encoding** for categorical features excluding the target variable.  
- Separate features (`X`) and target (`y`) for modeling.  
- Split data into training and testing sets using `train_test_split`.

### 3. Data Storage
- Save training and testing sets as CSV files for future use:
  - `./Training_Data/X_train.csv`  
  - `./Training_Data/y_train.csv`  
  - `./Testing_Data/X_test.csv`  
  - `./Testing_Data/y_test.csv`

### 4. Clustering Analysis
- Identify optimal number of clusters using the **elbow method**.  
- Train a **K-Means clustering model** on the dataset.  
- Visualize and label resulting clusters to gain actionable insights.

## Libraries Used
- Python 3.x  
- Pandas  
- NumPy  
- Scikit-learn (`train_test_split`, `KMeans`)  
- Matplotlib  
- OS, Sys  

## Purpose
The project ensures that the dataset is clean, processed, and ready for machine learning. It provides insights into customer behavior and creates a framework for predicting churn effectively, helping businesses make informed retention strategies.

## Authors
Sishir Pandey - Project Manager

Fahim Arman - Data Engineer

Jitesh Akaveeti - Data Analyst ( Predictive Modelling)

Chen - Data Analyst (Clustering)

Preeti Khatri - Data Analyst (Predictive Modelling)

Bishesh Aryal - Business Analyst


