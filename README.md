# 2024_ia651_Darimireddy_vaduguru

# Global Superstore Data Analysis and Predictive Modeling

## Introduction

This project involves data exploration, preprocessing, and predictive modeling using a dataset from a global superstore. The dataset includes information such as sales, discounts, profits, shipping costs, order dates, and various categorical variables. The primary objectives are:

1. **Exploratory Data Analysis (EDA):** Understanding the data structure and relationships between features.
2. **Data Preprocessing:** Handling missing values, transforming skewed data, and encoding categorical variables.
3. **Predictive Modeling:**
   - Predicting profit based on other features.
   - Predicting order priority using logistic regression, decision trees, random forests, and support vector machines.

## Data Exploration and Preprocessing

### 1. Loading the Dataset

The dataset was loaded from an Excel file (`global-superstore.xlsx`). Initial inspection revealed:

- **Columns:** 24
- **Rows:** 51,290

### 2. Handling Missing Values

The dataset contained missing values, particularly in the `Postal Code` column. This column was dropped as it was not critical to the analysis.

- **Resulting Shape of the Dataset:**
  - **Columns:** 23
  - **Rows:** 51,290

### 3. Correlation Analysis

Correlation analysis was performed on numerical features:

- **Sales**
- **Quantity**
- **Discount**
- **Profit**
- **Shipping Cost**

A heatmap was plotted to visualize correlations:

- **Sales vs. Profit:** Positive correlation
- **Discount vs. Profit:** Negative correlation
- **Quantity vs. Sales:** Positive correlation

### 4. Categorical Data Summary

Categorical columns were summarized:

- **Ship mode**
- **Segment Distribution**
- **Market Distribution:**
- **Region Distribution:**
- **Category Distribution:**
- **Sub - Category Distribution:**
- **Order Priority Distribution:**

### 5. Skewness and Log Transformation on numerical variables

Skewness was computed for numerical features, to have a normal distutribution in the plots 
The plotting was done in comparative way with the plots of before skewed and after skewed
on the varibles like sales, profit and shipping cost

### 6. Feature Selection and Encoding

Features selected for further analysis:

- **Order Date**
- **Segment**
- **Region**
- **Sub-Category**
- **Sales**
- **Profit**
- **Discount**
- **Shipping Cost**
- **Order Priority**
- **Quantity**

Categorical variables were encoded using **One-Hot Encoding**.

### 7. Data Splitting

The dataset was split into training and testing sets (80-20 split):

- **Profit Prediction:**
  - **X (Features):** All columns except `Profit`.
  - **y (Target):** `Profit`
- **Order Priority Prediction:**
  - **X (Features):** All columns except `Order Priority`.
  - **y (Target):** `Order Priority`

### 8. Scaling and Transformation

Standardization was applied using `StandardScaler` to ensure features are on the same scale, important for algorithms like SVM.

## Predictive Modeling: Profit Prediction

### 1. Region-Based Linear Regression

Profit prediction models were built separately for each region:

- **East Region**
  - **MSE:** 0.0985
  - **R^2 Score:** 0.8394
  - **Data Points:** 676
- **Oceania Region:**
  - **MSE:** 0.1971
  - **R^2 Score:** 0.6867
  - **Data Points:** 726
- **Africa Region:**
  - **MSE:** 0.145
  - **R^2 Score:** 0.6972
  - **Data Points:** 840
- **West Region:**
  - **MSE:** 0.0845
  - **R^2 Score:** 0.796
  - **Data Points:** 755
- **South Region:**
  - **MSE:** 0.1141
  - **R^2 Score:** 0.7368
  - **Data Points:** 1023
- **Central Asia Region:**
  - **MSE:** 0.1423
  - **R^2 Score:** 0.7318
  - **Data Points:** 564
- **EMEA Region:**
  - **MSE:** 0.1645
  - **R^2 Score:** 0.6711
  - **Data Points:** 830
- **North Asia Region:**
  - **MSE:** 0.1924
  - **R^2 Score:** 0.6867
  - **Data Points:** 637
- **Central Region:**
  - **MSE:** 0.0983
  - **R^2 Score:** 0.7218
  - **Data Points:** 1184
- **North Region:**
  - **MSE:** 0.1798
  - **R^2 Score:** 0.65
  - **Data Points:** 919
- **Canada Region:**
  - **MSE:** 0.2056
  - **R^2 Score:** 0.67
  - **Data Points:** 161
- **Southeast Asia Region:**
  - **MSE:** 0.21
  - **R^2 Score:** 0.74
  - **Data Points:** 585
- **Caribbean Region:**
  - **MSE:** 0.2144
  - **R^2 Score:** 0.667
  - **Data Points:** 465



- **East Region**
  - **MSE:** 0.2015
  - **R^2 Score:** 0.7706
- **Oceania Region:**
  - **MSE:** 0.2457
  - **R^2 Score:** 0.7506
- **Africa Region:**
  - **MSE:** 0.2364
  - **R^2 Score:** 0.6972
- **West Region:**
  - **MSE:** 0.2422
  - **R^2 Score:** 0.6464
- **South Region:**
  - **MSE:** 0.2687
  - **R^2 Score:** 0.6188
- **Central Asia Region:**
  - **MSE:** 0.2791
  - **R^2 Score:** 0.6903 
- **EMEA Region:**
  - **MSE:** 0.2698
  - **R^2 Score:** 0.6621
- **North Asia Region:**
  - **MSE:** 0.2568
  - **R^2 Score:** 0.6946
- **Central Region:**
  - **MSE:** 0.2291
  - **R^2 Score:** 0.6329
- **North Region:**
  - **MSE:** 0.3056
  - **R^2 Score:** 0.6268
- **Canada Region:**
  - **MSE:** 0.3795
  - **R^2 Score:** 0.5663
- **Southeast Asia Region:**
  - **MSE:** 0.295
  - **R^2 Score:** 0.7062
- **Caribbean Region:**
  - **MSE:** 0.1746
  - **R^2 Score:** 0.7556




### 2. Testing the Model

Test set performance was consistent with training:

- **Central Region (Testing Set):**
  - **MSE:** 0.0038
  - **R^2 Score:** 0.83
- **East Region (Testing Set):**
  - **MSE:** 0.0045
  - **R^2 Score:** 0.79
- **West Region (Testing Set):**
  - **MSE:** 0.0041
  - **R^2 Score:** 0.82
- **South Region (Testing Set):**
  - **MSE:** 0.0053
  - **R^2 Score:** 0.75

## Predictive Modeling: Order Priority Prediction

### 1. Logistic Regression

Logistic regression was used to predict `Order Priority`.

- **Training Accuracy:** 0.73
- **Training F1 Score (Macro Average):** 0.51
- **Testing Accuracy:** 0.73
- **Testing F1 Score (Macro Average):** 0.50

### 2. Decision Tree Classifier

A decision tree classifier was used to predict `Order Priority`.

- **Training Accuracy:** 0.95
- **Training F1 Score (Macro Average):** 0.93
- **Testing Accuracy:** 0.55
- **Testing F1 Score (Macro Average):** 0.48

### 3. Random Forest Classifier

A random forest classifier was trained to improve performance.

- **Training Accuracy:** 0.95
- **Training F1 Score (Macro Average):** 0.92
- **Testing Accuracy:** 0.59
- **Testing F1 Score (Macro Average):** 0.52

### 4. Support Vector Machine (SVM)

An SVM model with an RBF kernel was trained.

- **Best Parameters (GridSearchCV):**
  - **C:** 100
  - **gamma:** 1
- **Training Accuracy:** 0.62
- **Training F1 Score (Macro Average):** 0.57
- **Testing Accuracy:** 0.60
- **Testing F1 Score (Macro Average):** 0.55

## Conclusion

### 1. Profit Prediction

- Region-based linear regression models performed well, particularly in the Central and West regions, with R^2 scores above 0.80 and low MSE values.

### 2. Order Priority Prediction

- The SVM model with optimized parameters performed best in balancing bias and variance, achieving the highest test accuracy and F1 score among the models tested.

