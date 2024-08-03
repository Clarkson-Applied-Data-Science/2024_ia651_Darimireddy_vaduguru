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

Cumulative data up until that day were considered for the model 
Profit prediction models were built separately for each region:
Model predictions on the training data:
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


Model predictions on the testing data:

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
The data of only the previous day were considered for building the model
Test set performance was consistent with training:


- **East Region**
  - **MSE:** 0.578
  - **R^2 Score:** 0.0621
- **Oceania Region:**
  - **MSE:** 0.5592
  - **R^2 Score:** 0.0372
- **Africa Region:**
  - **MSE:** 0.4354
  - **R^2 Score:** 0.061
- **West Region:**
  - **MSE:** 0.3882
  - **R^2 Score:** 0.0452
- **South Region:**
  - **MSE:** 0.4146
  - **R^2 Score:** 0.0209
- **Central Asia Region:**
  - **MSE:** 0.5471
  - **R^2 Score:** 0.048 
- **EMEA Region:**
  - **MSE:** 0.471
  - **R^2 Score:** 0.0415
- **North Asia Region:**
  - **MSE:** 0.6007
  - **R^2 Score:** 0.050
- **Central Region:**
  - **MSE:** 0.3344
  - **R^2 Score:** 0.0247
- **North Region:**
  - **MSE:** 0.4611
  - **R^2 Score:** 0.033
- **Canada Region:**
  - **MSE:** 9.48563070798958e-30
  - **R^2 Score:** 1.0
- **Southeast Asia Region:**
  - **MSE:** 0.857
  - **R^2 Score:** 0.0613
- **Caribbean Region:**
  - **MSE:** 1.136
  - **R^2 Score:** -0.7006


- **East Region**
  - **MSE:** 1.493
  - **R^2 Score:** -0.6874
- **Oceania Region:**
  - **MSE:** 1.2421
  - **R^2 Score:** -0.7053
- **Africa Region:**
  - **MSE:** 1.8289
  - **R^2 Score:** -1.1398
- **West Region:**
  - **MSE:** 1.2065
  - **R^2 Score:** -0.9846
- **South Region:**
  - **MSE:** 1.2871
  - **R^2 Score:** -0.9764
- **Central Asia Region:**
  - **MSE:** 1.5318
  - **R^2 Score:** -0.4252 
- **EMEA Region:**
  - **MSE:** 1.2069
  - **R^2 Score:** -0.7217
- **North Asia Region:**
  - **MSE:** 2.030
  - **R^2 Score:** -1.737
- **Central Region:**
  - **MSE:** 1.4704
  - **R^2 Score:** -1.3752
- **North Region:**
  - **MSE:**  1.5409
  - **R^2 Score:** -1.0061
- **Canada Region:**
  - **MSE:** 1.4112
  - **R^2 Score:** 0.0755
- **Southeast Asia Region:**
  - **MSE:** 1.1205
  - **R^2 Score:** -0.3299
- **Caribbean Region:**
  - **MSE:** 1.1673
  - **R^2 Score:** -0.4571


## Predictive Modeling: Order Priority Prediction

### 1. Logistic Regression

Logistic regression was used to predict `Order Priority`.

- **Training Accuracy:** 0.7299
- **Training F1 Score (Macro Average):** 0.51
- **Testing Accuracy:** 0.7337
- **Testing F1 Score (Macro Average):** 0.50

### 2. Decision Tree Classifier

A decision tree classifier was used to predict `Order Priority`.

- **Testing Accuracy:** 0.5992
- **Testing F1 Score (Macro Average):** 0.43

### 3. Random Forest Classifier

A random forest classifier was trained to improve performance.

- **Training Accuracy:** 1.0
- **Training F1 Score (Macro Average):** 1.0
- **Testing Accuracy:** 0.69
- **Testing F1 Score (Macro Average):** 0.43

### 4. Support Vector Machine (SVM)

An SVM model with an RBF kernel was trained.

- **Best Parameters (GridSearchCV):**
  - **Best Score:** 0.4463
  - **C:** 100
  - **gamma:** 1
- **Training Accuracy:** 0.99
- **Training F1 Score (Macro Average):** 0.99
- **Testing Accuracy:** 0.585
- **Testing F1 Score (Macro Average):** 0.36

## Conclusion

### 1. Profit Prediction

- Region-based linear regression models performed well, particularly in the Central and West regions, with R^2 scores above 0.80 and low MSE values.

### 2. Order Priority Prediction

- The SVM model with optimized parameters performed best in balancing bias and variance, achieving the highest test accuracy and F1 score among the models tested.

<<image src="1.png">>
<<image src="2.png">>
<<image src="3.png">>
<<image src="4.png">>
<<image src="5.png">>
<<image src="6.png">>
<<image src="7.png">>
<<image src="8.png">>
<<image src="9.png">>
<<image src="10.png">>
<<image src="11.png">>
