# Customer Churn Analysis and Prediction

## Project Overview
This project aims to analyze customer churn behavior and prepare the dataset for machine learning modeling. By exploring customer demographics, service usage, and contract types, the project provides insights into patterns that lead to churn and sets the foundation for predictive modeling.

## Folder Structure

```
📁CHURN-PREDICTION
└── 📁Clustering_Analysis
|   └── 📁Clustering Analysis Documentation
|   |   ├── Clustering_Analysis.docx
|   |   ├── Clustering_Analysis.pdf
|   └── 📁data
|   |   ├── X_train.csv
|   └── 📁results
|   |   ├── cluster_center.xlsx
|   |   ├── cluster_label.xlsx
|   |   ├── Cluster_scatter_plot.png
|   |   ├── Clustering_results.xlsx
|   |   ├── Elbow.png
|   ├── clustering_analysis.ipynb
└── 📁Data_Preparation
|   └── 📁Preprocessed_Data
|   |   ├── preprocessed_data_with_encoding_categorical.csv
|   |   ├── preprocessed_dataset.csv
|   └── 📁Scaling Techniques Documentation
|   |   ├── Data_Preparation.docx
|   |   ├── Data_Preparation.pdf
|   └── 📁Testing_Data
|   |   ├── X_test.csv
|   |   ├── y_test.csv
|   └── 📁Training_Data
|   |   ├── X_train.csv
|   |   ├── y_train.csv
|   ├── data_preparation.ipynb
└── README.md
```

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

## Scaling Techniques Documentation

The project ensures that the dataset is clean, processed, and ready for machine learning. It provides insights into customer behavior and creates a framework for predicting churn effectively, helping businesses make informed retention strategies.

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
  - `./Preprocessed_Data/preprocessed_data_with_encoding_categorical.csv`  
  - `./Training_Data/X_train.csv`  
  - `./Training_Data/y_train.csv`  
  - `./Testing_Data/X_test.csv`  
  - `./Testing_Data/y_test.csv`


## Clustering Analysis Documentation

### 1. Import Section
- Imports essential Python libraries for data preprocessing, clustering, and visualization:
  - **Pandas**: Load and manage datasets.
  - **NumPy**: Numerical computations and array operations.
  - **Scikit-learn (KMeans)**: Perform customer segmentation.
  - **PCA**: Dimensionality reduction for visualization.
  - **Matplotlib & SciPy**: Visualize clusters and draw boundaries.
  - **matplotlib.cm**: Assign distinct colors to clusters for interpretation.


### 2. Data Loading
- Dataset is read from `X_train.csv` using `pd.read_csv()`.  
- Stored in a Pandas DataFrame (`data`) for easy manipulation.  

### 3. Data View
- `data.head()` displays the first 5 rows to inspect structure, column names, and sample values.  

### 4. Data Normalization
- Converts DataFrame to NumPy array for faster computation.  
- Standardizes data (mean = 0, std = 1) to improve model performance.  
- PCA reduces data to **2 principal components** for easier visualization (`X_pca`).  

### 5. Elbow Method
- Determines the **optimal number of clusters (k)** for K-Means.  
- Steps:
  - Test k from 1 to 15, compute inertia (within-cluster sum of squares).  
  - Plot k vs inertia to identify the “elbow point.”  
- **Result**: Optimal k = 4.  
- Plot saved as `Elbow.png`.  


### 6. Optimal Number of Clusters
- Uses Elbow Method to find point where additional clusters no longer improve model performance.  
- K = 4 chosen for best balance between simplicity and accuracy.  
- Visualization shows inertia reduction from K=1 to K=4, flattening afterward.  


### 7. Clustering Analysis
- **Number of clusters**: K = 4.  
- **Steps:**
  - Initialize and fit KMeans on PCA-transformed data.  
  - Store cluster labels and centroids.  
  - Visualize clusters with distinct colors, Convex Hull boundaries, and centroids.  
  - Add legends, axis labels, and a title.  
  - Save plot as `Cluster_scatter_plot.png`.  
- Visual output shows four distinct customer segments.

### 8. Save Data
- Reconstruct cluster centers to original feature space using `inverse_transform()`.  
- Rescale centers with original data mean and std.  
- Save outputs:
  - **Cluster Centers**: `cluster_center.xlsx`  
  - **Cluster Labels**: `cluster_label.xlsx`  
  - **Cluster Summary Report**: `Clustering_results.xlsx`  
    - Includes cluster name, frequency, and percentage of total customers.


## Results
- **Optimal Clusters**: 4  
- **Outputs Generated**:
  - `Elbow.png` → Elbow Method plot  
  - `Cluster_scatter_plot.png` → Cluster visualization  
  - `cluster_center.xlsx` → Cluster centroids  
  - `cluster_label.xlsx` → Customer cluster assignments  
  - `Clustering_results.xlsx` → Summary report  


## Conclusion
This clustering analysis provides a **clear segmentation of customers**, which can be used to:
- Identify high-risk churn groups  
- Develop targeted marketing strategies  
- Support predictive modeling for customer retention  

---

## Libraries Used
- Python 3.x  
- Pandas  
- NumPy  
- Scikit-learn (`train_test_split`, `KMeans`, `PCA`)  
- Matplotlib  
- OS, Sys  


## Authors
Sishir Pandey - Project Manager

Fahim Arman - Data Engineer

Chen - Data Analyst (Clustering)

Jitesh Akaveeti - Data Analyst ( Predictive Modelling)

Preeti Khatri - Data Analyst (Predictive Modelling)

Bishesh Aryal - Business Analyst


