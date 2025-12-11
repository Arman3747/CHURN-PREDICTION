# Customer Churn Analysis and Prediction

## Project Overview

This project aims to predict customer churn and segment customers for a telecommunications company. By analyzing customer data, performing clustering, and applying predictive modeling techniques, the team identified high-risk customer groups, key churn drivers, and actionable retention strategies. The project combines unsupervised learning (K-Means clustering) with supervised learning (Artificial Neural Networks) to provide both descriptive and predictive insights.

## Table of Contents

- [Project Overview](#project-overview)
- [Data Description](#data-description)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Clustering Analysis](#clustering-analysis)
  - [Predictive Modeling](#predictive-modeling)
- [Key Findings](#key-findings)
- [Retention Strategies](#retention-strategies)
- [Limitations & Future Improvements](#limitations--future-improvements)
- [Conclusion](#conclusion)
- [Directory Structure](#directory-structure)
- [Libraries Used](#libraries-used)
- [Reproducibility](#reproducibility)

## Data Description

The dataset consists of **7,043 customer records** with the following features:

- **Demographic**: `Gender`, `SeniorCitizen`, `Dependents`
- **Account Info**: `Tenure`, `Contract`, `MonthlyCharges`
- **Services**: `PhoneService`, `MultipleLines`, `InternetService`
- **Target Variable**: `Churn` (`Yes`/`No`)

The dataset is complete with no missing values, and the target variable is imbalanced: 1,869 churned vs 5,174 non-churned customers.

<!-- <details>
<summary>Data Description</summary>

The dataset consists of **7,043 customer records** with the following features:

- **Demographic**: `Gender`, `SeniorCitizen`, `Dependents`
- **Account Info**: `Tenure`, `Contract`, `MonthlyCharges`
- **Services**: `PhoneService`, `MultipleLines`, `InternetService`
- **Target Variable**: `Churn` (`Yes`/`No`)

The dataset is complete with no missing values, and the target variable is imbalanced: 1,869 churned vs 5,174 non-churned customers.

</details> -->

## Methodology

### Data Preprocessing

- Loaded data using Pandas and inspected the structure (`head()`, `info()`, `shape`).
- Converted categorical columns to numeric using **Label Encoding**.
- Scaled numerical features with **StandardScaler** for uniformity.
- Split the dataset into training (80%) and testing (20%) sets with stratification to maintain class distribution.
- Applied **one-hot encoding** to categorical features for machine learning compatibility.

### Clustering Analysis

- **Normalization**: Standardized features to mean 0, standard deviation 1.
- **Dimensionality Reduction**: Applied PCA to reduce to 2 components for visualization.
- **K-Means Clustering**:
  - Optimal clusters determined using the **Elbow Method** → `K = 4`.
  - Visualized clusters with convex hulls and plotted centroids.
  - Saved cluster labels and centers for further analysis.

### Predictive Modeling (ANN)

- Built an **Artificial Neural Network (ANN)** using Keras:
  - Input layer = number of features
  - Hidden layers = 64 and 32 neurons with ReLU and Dropout (0.3)
  - Output layer = 1 neuron with Sigmoid activation for binary classification
- Trained for 50 epochs, batch size 32, with 20% validation split.
- Evaluated performance using:
  - **Accuracy**: ~78%
  - **ROC AUC**: 0.66
  - **Confusion Matrix**: better performance for non-churners than churners
- Saved trained model as `ANN_Churn_Model.h5` for future predictions.

## Key Findings

- **Customer Segmentation**: K-Means clustering revealed 3 meaningful groups:

  1. Loyal and stable customers
  2. Value-sensitive customers
  3. High-risk new customers

- **Churn Drivers Identified**:

  - Contract type (month-to-month has higher churn)
  - Customer tenure (newer customers are at higher risk)
  - Monthly charges (higher charges increase churn probability)
  - Usage of support services and add-ons

- **ANN Insights**: Confirmed churn patterns based on service engagement, billing levels, and contract duration.

## Retention Strategies

### Segment-Specific

- **Loyal & Stable Customers**: Loyalty rewards, maintain service quality, request feedback.
- **Value-Sensitive Customers**: Targeted discounts, mid-tier bundle offers.
- **High-Risk New Customers**: Early engagement campaigns, short-term incentives, encourage long-term contracts.

### Model-Driven

- Flag high-risk customers using churn probabilities.
- Tailor retention offers based on key drivers (e.g., billing, service usage).
- Integrate predictions into CRM for proactive interventions.
- Automate retention workflows and monitor metrics continuously.

## Limitations & Future Improvements

- **Class Imbalance**: Fewer churned customers → risk of biased predictions.
- **Lack of Qualitative Data**: Missing customer satisfaction, complaints, or support interactions.
- **Model Interpretability**: Complex ANN models may be harder for stakeholders to understand.
- **Static Data**: Dataset is a snapshot; evolving behaviors may affect model accuracy.

**Proposed Solutions**:

- Use SMOTE or class weighting to handle imbalance.
- Apply SHAP or feature importance for interpretability.
- Continuously update and retrain models as new data arrives.
- Monitor performance over time and optimize hyperparameters.

## Conclusion

This project successfully identified high-risk customers and provided actionable insights into customer churn. Combining clustering with supervised learning allows the company to move from reactive retention to proactive strategies. The findings support informed decision-making, customer loyalty enhancement, and long-term value maximization.

## Directory Structure

```
📁Clustering_Analysis
├── 📁Clustering Analysis Documentation
|   |   ├── Clustering_Analysis.docx
|   |   ├── Clustering_Analysis.pdf
|   ├──📁data
|   |   ├── X_train.csv
|   ├── 📁results
|   |   ├── cluster_center.xlsx
|   |   ├── cluster_label.xlsx
|   |   ├── Cluster_scatter_plot.png
|   |   ├── Clustering_results.xlsx
|   |   ├── Elbow.png
|   ├── clustering_analysis.ipynb
├── 📁Data_Preparation
|   ├── 📁Preprocessed_Data
|   |   ├── preprocessed_data_with_encoding_categorical.csv
|   |   ├── preprocessed_dataset.csv
|   ├── 📁Scaling Techniques Documentation
|   |   ├── Data_Preparation.docx
|   |   ├── Data_Preparation.pdf
|   ├── 📁Testing_Data
|   |   ├── X_test.csv
|   |   ├── y_test.csv
|   ├── 📁Training_Data
|   |   ├── X_train.csv
|   |   ├── y_train.csv
|   ├── data_preparation.ipynb
├── 📁Final_Report
|   ├── Final_Report.docx
|   ├── Final_Report.pdf
├── 📁Predictive_Modeling
|   ├── 📁Predictive Modeling Documentation
|   |   ├── Predictive_Modeling.docx
|   |   ├── Predictive_Modeling.pdf
|   ├── 📁results
|   |   ├── ANN_Churn_Model.h5
|   |   ├── Confusion_Matrix.png
|   |   ├── ROC.png
|   ├── Predictive_Analysis.ipynb
└── README.md
```

## Libraries Used

- Python 3.x
- Pandas
- NumPy
- Scipy (`ConvexHull`)
- Scikit-learn (`LabelEncoder`, `StandardScaler`, `train_test_split`, `KMeans`, `PCA`, `classification_report`, `confusion_matrix`, `accuracy_score`, `roc_auc_score`, `roc_curve`)
- tensorflow (`Sequential`, `Dense`, `Dropout`)
- Matplotlib
- seaborn
- OS, Sys

## Reproducibility

Follow the steps below to set up the project locally:

```bash
# 1. Clone the repository
git clone https://github.com/Arman3747/CHURN-PREDICTION.git
cd CHURN-PREDICTION

# 2. Install dependencies
pip install -r requirements.txt
```

## Project Team:

- Sishir Pandey - Project Manager - [LinkedIn](https://www.linkedin.com/in/connectsishir/)
- Fahim Arman - Data Engineer - [LinkedIn](https://www.linkedin.com/in/fahim37/)
- Chen - Data Analyst (Clustering)
- Jitesh Akaveeti - Data Analyst (Predictive Modelling)
- Preeti Khatri - Data Analyst (Predictive Modelling)
- Bishesh Aryal - Business Analyst

---

Thanks For Reading !!!
