# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import metrics
from sklearn.svm import SVC

# %%
file_path = 'global-superstore.xlsx'
df = pd.read_excel(file_path)
df.head(5)

# %%
print("Columns: {}\nRows: {}".format(df.shape[1],df.shape[0]))

# %%
df.isnull().sum()

# %%
df = df.drop(columns=['Postal Code'])

# %%
print("Columns: {}\nRows: {}".format(df.shape[1],df.shape[0]))

# %%
numeric_columns = ['Sales','Quantity','Discount','Profit','Shipping Cost']
correlation_matrix = df[numeric_columns].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# %%
cat_cols = [col for col in df.columns if df[col].dtype == "object" and df[col].nunique() <= 20]
def cat_summary(df, col, plot=False):
    if plot:
        plt.figure(figsize=(10, 8))
        sns.countplot(x=col, data=df, hue=col, palette='viridis', legend=False)
        plt.xticks(rotation=60)
        plt.title(f'Count Plot of {col}')
        plt.show()
    else:
        summary = df[col].value_counts()
        print(f"Summary for {col}:\n{summary}\n")

for col in cat_cols:
    cat_summary(df, col, plot=True)

# %%
numerical_data = ['Sales', 'Profit', 'Shipping Cost']

skewness = df[numerical_data].skew()

df_log_transformed = df.copy()
for feature in numerical_data:
    if skewness[feature] > 0.5:  
        df_log_transformed[feature] = np.log1p(df_log_transformed[feature])
        
for feature in numerical_data:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.histplot(df[feature], bins=100, kde=True, ax=axes[0])
    axes[0].set_title(f'Original Distribution of {feature}')
    if skewness[feature] > 0.5:
        sns.histplot(df_log_transformed[feature], bins=100, kde=True, ax=axes[1])
        axes[1].set_title(f'Log-Transformed Distribution of {feature}')
    else:
        axes[1].set_visible(False)

    plt.tight_layout()

    plt.show()

# %%
columns_to_select = ['Order Date', 'Segment', 'Region', 'Sub-Category','Sales', 'Profit', 'Discount','Shipping Cost','Order Priority','Quantity'] 
df_selected = df[columns_to_select]
df_selected.head()

# %%
df_selected.isnull().sum()

# %%
encoder = OneHotEncoder()
encoded_array = encoder.fit_transform(df_selected[['Segment','Sub-Category','Order Priority']]).toarray()
encoded_df_selected = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['Segment','Sub-Category','Order Priority']), index=df_selected.index)
encoded_df_selected.head()

# %%
region_values = df['Region'].unique()
print(region_values)

# %%
df_combined = pd.concat([df_selected,encoded_df_selected], axis=1)
df_combined.head()

# %%
df_combined = df_combined.drop(columns=['Segment','Sub-Category','Order Priority'])
df_combined.head()

# %%
df_combined = df_combined[df_combined['Profit']>0]

# %%
X = df_combined.drop(columns=['Profit'])
y = df_combined[['Order Date','Region','Profit']]

# %%
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# %%
X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

# %%
X_train_old = X_train.drop(columns=['Sales', 'Shipping Cost'])
X_test_old = X_test.drop(columns=['Sales', 'Shipping Cost'])

# %%
y_train_old = y_train.drop(columns=['Profit'])
y_test_old = y_test.drop(columns=['Profit'])


# %%
y_train.describe()

# %%
y_test.describe()

# %%
X_train_new = X_train[['Sales', 'Shipping Cost']]
X_test_new = X_test[['Sales', 'Shipping Cost']]

y_train_new = pd.DataFrame(y_train, columns=['Profit'])
y_test_new = pd.DataFrame(y_test, columns=['Profit'])

X_train_log = np.log(X_train_new)
X_test_log = np.log(X_test_new)

y_train_log = np.log(y_train_new)
y_test_log = np.log(y_test_new)


# %%
X_train_log = X_train_log.reset_index(drop=True)
X_train_old = X_train_old.reset_index(drop=True)

X_test_log = X_test_log.reset_index(drop=True)
X_test_old = X_test_old.reset_index(drop=True)

y_train_log = y_train_log.reset_index(drop=True)
y_train_old = y_train_old.reset_index(drop=True)

y_test_log = y_test_log.reset_index(drop=True)
y_test_old = y_test_old.reset_index(drop=True)

# %%
X_train_log = pd.concat([X_train_log,X_train_old[['Discount','Quantity']]], axis = 1)
X_test_log = pd.concat([X_test_log,X_test_old[['Discount','Quantity']]], axis = 1)

# %%
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_log)

X_test_scaled = scaler.transform(X_test_log)

y_train_scaled = scaler.fit_transform(y_train_log)

y_test_scaled = scaler.transform(y_test_log)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=['Sales', 'Shipping Cost','Discount','Quantity'])
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=['Sales', 'Shipping Cost','Discount','Quantity'])
y_train_scaled_df = pd.DataFrame(y_train_scaled, columns=['Profit'])
y_test_scaled_df = pd.DataFrame(y_test_scaled, columns=['Profit'])


# %%
X_train_scaled_df = X_train_scaled_df.reset_index(drop=True)
X_train_old = X_train_old.reset_index(drop=True)

X_test_scaled_df = X_test_scaled_df.reset_index(drop=True)
X_test_old = X_test_old.reset_index(drop=True)

y_train_scaled_df = y_train_scaled_df.reset_index(drop=True)
y_train_old = y_train_old.reset_index(drop=True)

y_test_scaled_df = y_test_scaled_df.reset_index(drop=True)
y_test_old = y_test_old.reset_index(drop=True)

# %%
X_train_scaled_df = pd.concat([X_train_scaled_df, X_train_old.drop(columns=['Discount','Quantity'])], axis=1)
X_test_scaled_df = pd.concat([X_test_scaled_df, X_test_old.drop(columns=['Discount','Quantity'])], axis=1)

# %%
y_train_scaled_df = pd.concat([y_train_scaled_df, y_train_old], axis=1)
y_test_scaled_df = pd.concat([y_test_scaled_df, y_test_old], axis=1)

# %%
X_train_scaled_df

# %%
df_combined['Region'].unique()

# %%
from sklearn.metrics import mean_squared_error, r2_score

for region in df_combined['Region'].unique():
        X_train_region = X_train_scaled_df[X_train_scaled_df['Region'] == region]
        y_train_region = y_train_scaled_df[y_train_scaled_df['Region'] == region]

        aggregated_data_y_train = y_train_region.groupby('Order Date').agg({
        'Profit': 'mean',
        }).reset_index()

        aggregated_data_X_train = X_train_region.groupby('Order Date').agg({
        'Sales': 'mean',
        'Quantity': 'mean',
        'Segment_Consumer': 'sum',
        'Segment_Corporate': 'sum',
        'Segment_Home Office': 'sum',
        'Sub-Category_Accessories': 'sum',
        'Sub-Category_Appliances': 'sum',
        'Sub-Category_Art': 'sum',
        'Sub-Category_Binders': 'sum',
        'Sub-Category_Bookcases': 'sum',
        'Sub-Category_Tables': 'sum',
        'Sub-Category_Chairs': 'sum',
        'Sub-Category_Phones': 'sum',
        'Sub-Category_Copiers': 'sum',
        'Sub-Category_Supplies': 'sum',
        'Sub-Category_Machines': 'sum',
        'Sub-Category_Storage': 'sum',
        'Sub-Category_Furnishings': 'sum',
        'Sub-Category_Paper': 'sum',
        'Sub-Category_Envelopes': 'sum',
        'Sub-Category_Fasteners': 'sum',
        'Sub-Category_Labels': 'sum',
        'Order Priority_Critical': 'sum',
        'Order Priority_High': 'sum',
        'Order Priority_Low': 'sum',
        'Order Priority_Medium': 'sum'
        }).reset_index()

        aggregated_data_X_train = aggregated_data_X_train.drop(columns=['Order Date'])
        aggregated_data_y_train = aggregated_data_y_train.drop(columns=['Order Date'])

        linear = LinearRegression()
        reg_train = linear.fit(aggregated_data_X_train , aggregated_data_y_train)
        y_pred = linear.predict(aggregated_data_X_train)
        mse = mean_squared_error(aggregated_data_y_train, y_pred)
        r2 = r2_score(aggregated_data_y_train, y_pred)
        print(f"For {region} Mean Squared Error (MSE): {mse}")
        print(f"R^2 Score: {r2}")
        num_rows = aggregated_data_X_train.shape[0]
        print(num_rows)


# %%
for region in df_combined['Region'].unique():
        X_test_region = X_test_scaled_df[X_test_scaled_df['Region'] == region]
        y_test_region = y_test_scaled_df[y_test_scaled_df['Region'] == region]

        aggregated_data_y_test = y_test_region.groupby('Order Date').agg({
        'Profit': 'mean',
        }).reset_index()

        aggregated_data_X_test = X_test_region.groupby('Order Date').agg({
        'Sales': 'mean',
        'Quantity': 'mean',
        'Segment_Consumer': 'sum',
        'Segment_Corporate': 'sum',
        'Segment_Home Office': 'sum',
        'Sub-Category_Accessories': 'sum',
        'Sub-Category_Appliances': 'sum',
        'Sub-Category_Art': 'sum',
        'Sub-Category_Binders': 'sum',
        'Sub-Category_Bookcases': 'sum',
        'Sub-Category_Tables': 'sum',
        'Sub-Category_Chairs': 'sum',
        'Sub-Category_Phones': 'sum',
        'Sub-Category_Copiers': 'sum',
        'Sub-Category_Supplies': 'sum',
        'Sub-Category_Machines': 'sum',
        'Sub-Category_Storage': 'sum',
        'Sub-Category_Furnishings': 'sum',
        'Sub-Category_Paper': 'sum',
        'Sub-Category_Envelopes': 'sum',
        'Sub-Category_Fasteners': 'sum',
        'Sub-Category_Labels': 'sum',
        'Order Priority_Critical': 'sum',
        'Order Priority_High': 'sum',
        'Order Priority_Low': 'sum',
        'Order Priority_Medium': 'sum'
    }).reset_index()

        aggregated_data_X_test = aggregated_data_X_test.drop(columns=['Order Date'])
        aggregated_data_y_test = aggregated_data_y_test.drop(columns=['Order Date'])

        linear = LinearRegression()
        reg_train = linear.fit(aggregated_data_X_train , aggregated_data_y_train)
        y_pred = linear.predict(aggregated_data_X_test)
        mse = mean_squared_error(aggregated_data_y_test, y_pred)
        r2 = r2_score(aggregated_data_y_test, y_pred)
        print(f"For region {region} Mean Squared Error (MSE): {mse}")
        print(f"R^2 Score: {r2}")


# %%

for region in df_combined['Region'].unique():
    # Filter the train data by region
    X_train_region = X_train_scaled_df[X_train_scaled_df['Region'] == region]
    y_train_region = y_train_scaled_df[y_train_scaled_df['Region'] == region]

    # Aggregate the data by date
    aggregated_data_y_train = y_train_region.groupby('Order Date').agg({
        'Profit': 'mean',
    }).reset_index()

    aggregated_data_X_train = X_train_region.groupby('Order Date').agg({
        'Sales': 'mean',
        'Quantity': 'mean',
        'Segment_Consumer': 'sum',
        'Segment_Corporate': 'sum',
        'Segment_Home Office': 'sum',
        'Sub-Category_Accessories': 'sum',
        'Sub-Category_Appliances': 'sum',
        'Sub-Category_Art': 'sum',
        'Sub-Category_Binders': 'sum',
        'Sub-Category_Bookcases': 'sum',
        'Sub-Category_Tables': 'sum',
        'Sub-Category_Chairs': 'sum',
        'Sub-Category_Phones': 'sum',
        'Sub-Category_Copiers': 'sum',
        'Sub-Category_Supplies': 'sum',
        'Sub-Category_Machines': 'sum',
        'Sub-Category_Storage': 'sum',
        'Sub-Category_Furnishings': 'sum',
        'Sub-Category_Paper': 'sum',
        'Sub-Category_Envelopes': 'sum',
        'Sub-Category_Fasteners': 'sum',
        'Sub-Category_Labels': 'sum',
        'Order Priority_Critical': 'sum',
        'Order Priority_High': 'sum',
        'Order Priority_Low': 'sum',
        'Order Priority_Medium': 'sum'
    }).reset_index()

    # Create lagged features by shifting the features by one day
    lagged_data_X_train = aggregated_data_X_train.copy()
    lagged_data_X_train['Order Date'] = lagged_data_X_train['Order Date'] + pd.DateOffset(days=1)

    # Rename lagged columns
    lagged_columns = {col: f'{col}_lag1' for col in lagged_data_X_train.columns if col != 'Order Date'}
    lagged_data_X_train = lagged_data_X_train.rename(columns=lagged_columns)

    # Merge the lagged data with the target variable
    merged_data_train = pd.merge(aggregated_data_y_train, lagged_data_X_train, on='Order Date', how='inner')

    # Drop rows with missing values after merging
    merged_data_train = merged_data_train.dropna()

    # Separate features and target
    X_train = merged_data_train.drop(columns=['Order Date', 'Profit'])
    y_train = merged_data_train['Profit']

    # Check if there's enough data to train the model
    if X_train.empty or y_train.empty:
        print(f"No data available for region {region} after processing.")
        continue

    # Train the model
    linear = LinearRegression()
    linear.fit(X_train, y_train)

    # Predict and evaluate the model
    y_pred = linear.predict(X_train)
    mse = mean_squared_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)

    print(f"For region {region} Mean Squared Error (MSE): {mse}")
    print(f"R^2 Score: {r2}")
    

# %%
for region in df_combined['Region'].unique():
    X_test_region = X_test_scaled_df[X_test_scaled_df['Region'] == region]
    y_test_region = y_test_scaled_df[y_test_scaled_df['Region'] == region]

    aggregated_data_y_test = y_test_region.groupby('Order Date').agg({
        'Profit': 'mean',
    }).reset_index()

    aggregated_data_X_test = X_test_region.groupby('Order Date').agg({
        'Sales': 'mean',
        'Quantity': 'mean',
        'Segment_Consumer': 'sum',
        'Segment_Corporate': 'sum',
        'Segment_Home Office': 'sum',
        'Sub-Category_Accessories': 'sum',
        'Sub-Category_Appliances': 'sum',
        'Sub-Category_Art': 'sum',
        'Sub-Category_Binders': 'sum',
        'Sub-Category_Bookcases': 'sum',
        'Sub-Category_Tables': 'sum',
        'Sub-Category_Chairs': 'sum',
        'Sub-Category_Phones': 'sum',
        'Sub-Category_Copiers': 'sum',
        'Sub-Category_Supplies': 'sum',
        'Sub-Category_Machines': 'sum',
        'Sub-Category_Storage': 'sum',
        'Sub-Category_Furnishings': 'sum',
        'Sub-Category_Paper': 'sum',
        'Sub-Category_Envelopes': 'sum',
        'Sub-Category_Fasteners': 'sum',
        'Sub-Category_Labels': 'sum',
        'Order Priority_Critical': 'sum',
        'Order Priority_High': 'sum',
        'Order Priority_Low': 'sum',
        'Order Priority_Medium': 'sum'
    }).reset_index()

    # Create lagged features by shifting the features by one day
    lagged_data_X_test = aggregated_data_X_test.copy()
    lagged_data_X_test['Order Date'] = lagged_data_X_test['Order Date'] + pd.DateOffset(days=1)

    # Rename lagged columns
    lagged_columns = {col: f'{col}_lag1' for col in lagged_data_X_test.columns if col != 'Order Date'}
    lagged_data_X_test = lagged_data_X_test.rename(columns=lagged_columns)

    # Merge the lagged data with the target variable
    merged_data_test = pd.merge(aggregated_data_y_test, lagged_data_X_test, on='Order Date', how='inner')

    # Drop rows with missing values after merging
    merged_data_test = merged_data_test.dropna()

    # Separate features and target
    X_test = merged_data_test.drop(columns=['Order Date', 'Profit'])
    y_test = merged_data_test['Profit']

    # Train the model
    linear = LinearRegression()
    linear.fit(X_train, y_train)

    # Predict and evaluate the model
    y_pred = linear.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"For region {region} Mean Squared Error (MSE): {mse}")
    print(f"R^2 Score: {r2}")


# %% [markdown]
# Logistic Regression

# %%
df_selected.head()

# %%
encoder = OneHotEncoder()
encoded_array = encoder.fit_transform(df_selected[['Segment','Sub-Category','Region']]).toarray()
encoded_df_selected = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['Segment','Sub-Category','Region']), index=df_selected.index)
encoded_df_selected.head()

# %%
df_combined1 = pd.concat([df_selected,encoded_df_selected], axis=1)

# %%
df_combined1 = df_combined1.drop(columns=['Segment','Sub-Category','Region'])

# %%
df_combined1 = df_combined1[df_combined1['Profit']>0]

# %%
df_combined1['Year'] = df_combined1['Order Date'].dt.year
df_combined1['Month'] = df_combined1['Order Date'].dt.month
df_combined1['Day'] = df_combined1['Order Date'].dt.day


# %%
df_combined1['Month_Sin'] = np.sin(2 * np.pi * df_combined1['Month'] / 12)
df_combined1['Month_Cos'] = np.cos(2 * np.pi * df_combined1['Month'] / 12)

# %%
df_combined1 = df_combined1.drop(columns = ['Year','Month','Day'])

# %%
X = df_combined1.drop(columns=['Order Priority'])
y = df_combined1[['Order Date','Order Priority']]

# %%
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# %%
X_train = pd.DataFrame(X_train, columns=X_train.columns)
X_test = pd.DataFrame(X_test, columns=X_test.columns)

# %%
y_train = pd.DataFrame(y_train, columns=y_train.columns)
y_test = pd.DataFrame(y_test, columns=y_test.columns)

# %%
X_train_old = X_train.drop(columns=['Sales', 'Shipping Cost','Profit'])
X_test_old = X_test.drop(columns=['Sales', 'Shipping Cost','Profit'])

# %%
X_train_new = X_train[['Sales', 'Shipping Cost','Profit']]
X_test_new = X_test[['Sales', 'Shipping Cost','Profit']]


X_train_log = np.log(X_train_new)
X_test_log = np.log(X_test_new)

# %%
X_train_log = X_train_log.reset_index(drop=True)
X_train_old = X_train_old.reset_index(drop=True)

X_test_log = X_test_log.reset_index(drop=True)
X_test_old = X_test_old.reset_index(drop=True)

# %%
X_train_log = pd.concat([X_train_log,X_train_old[['Discount','Quantity']]], axis = 1)
X_test_log = pd.concat([X_test_log,X_test_old[['Discount','Quantity']]], axis = 1)

# %%
X_test_log.describe()

# %%
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_log)

X_test_scaled = scaler.transform(X_test_log)


X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=['Sales', 'Shipping Cost','Discount','Quantity','Profit'])
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=['Sales', 'Shipping Cost','Discount','Quantity','Profit'])

# %%
X_train_scaled_df = X_train_scaled_df.reset_index(drop=True)
X_train_old = X_train_old.reset_index(drop=True)

X_test_scaled_df = X_test_scaled_df.reset_index(drop=True)
X_test_old = X_test_old.reset_index(drop=True)

# %%
X_train_scaled_df = pd.concat([X_train_scaled_df, X_train_old.drop(columns=['Discount','Quantity'])], axis=1)
X_test_scaled_df = pd.concat([X_test_scaled_df, X_test_old.drop(columns=['Discount','Quantity'])], axis=1)

# %%
X_train_scaled_df = X_train_scaled_df.drop(columns=['Order Date'])
X_test_scaled_df = X_test_scaled_df.drop(columns=['Order Date'])

# %%
model1 = LogisticRegression()
log = model1.fit(X_train_scaled_df, y_train['Order Priority'])

y_pred = log.predict(X_test_scaled_df)
y_pred_train = log.predict(X_train_scaled_df)

print(metrics.accuracy_score(y_train['Order Priority'], y_pred_train),
      metrics.f1_score(y_train['Order Priority'], y_pred_train, average='macro'))


print(metrics.accuracy_score(y_test['Order Priority'], y_pred),
      metrics.f1_score(y_test['Order Priority'], y_pred, average='macro'))

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

model3 = DecisionTreeClassifier()
dc = model3.fit(X_train_scaled_df, y_train['Order Priority'])

y_pred = dc.predict(X_test_scaled_df)
y_pred_train = dc.predict(X_train_scaled_df)

print(metrics.accuracy_score(y_train['Order Priority'], y_pred_train),
      metrics.f1_score(y_train['Order Priority'], y_pred_train, average='macro'))


print(metrics.accuracy_score(y_test['Order Priority'], y_pred),
      metrics.f1_score(y_test['Order Priority'], y_pred, average='macro'))

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

model4 = RandomForestClassifier()
rf = model4.fit(X_train_scaled_df, y_train['Order Priority'])

y_pred_train_rf = rf.predict(X_train_scaled_df)
y_pred_test_rf = rf.predict(X_test_scaled_df)

print("Random Forest - Training Accuracy:", metrics.accuracy_score(y_train['Order Priority'], y_pred_train_rf))
print("Random Forest - Training F1 Score:", metrics.f1_score(y_train['Order Priority'], y_pred_train_rf, average='macro'))
print("Random Forest - Test Accuracy:", metrics.accuracy_score(y_test['Order Priority'], y_pred_test_rf))
print("Random Forest - Test F1 Score:", metrics.f1_score(y_test['Order Priority'], y_pred_test_rf, average='macro'))

# %%
from sklearn.svm import  SVC
model2 = SVC(kernel = 'rbf', C=1000, gamma=1)
svm = model2.fit(X_train_scaled_df, y_train['Order Priority'])

y_pred = svm.predict(X_test_scaled_df)
y_pred_train = svm.predict(X_train_scaled_df)

print(metrics.accuracy_score(y_train['Order Priority'], y_pred_train),
      metrics.f1_score(y_train['Order Priority'], y_pred_train, average='macro'))


print(metrics.accuracy_score(y_test['Order Priority'], y_pred),
      metrics.f1_score(y_test['Order Priority'], y_pred, average='macro'))

# %%
from sklearn.model_selection import GridSearchCV, KFold
folds = KFold(n_splits=5, shuffle=True, random_state=7)
model = SVC()

params = {'C': [0.1, 1, 10, 100, 1000, 10000, 20000],
          'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}

c_opt = GridSearchCV(estimator=model, param_grid=params,
                     scoring='f1_macro', cv=folds, n_jobs=-1,
                     verbose=1, return_train_score=True)

c_opt.fit(X_train_scaled_df, y_train['Order Priority'])
c_results = pd.DataFrame(c_opt.cv_results_)
c_results

# %%
print(f'Best Score: {0.446347}')
print('Best Parameters: {"C": 100, "gamma": 1}')

# %%
from sklearn.svm import  SVC
model2 = SVC(kernel = 'rbf', C=100, gamma=1)
model2.fit(X_train_scaled_df, y_train['Order Priority'])

y_pred = model2.predict(X_test_scaled_df)
y_pred_train = model2.predict(X_train_scaled_df)

print(metrics.accuracy_score(y_train['Order Priority'], y_pred_train),
      metrics.f1_score(y_test['Order Priority'], y_pred, average='macro'))


print(metrics.accuracy_score(y_test['Order Priority'], y_pred),
      metrics.f1_score(y_test['Order Priority'], y_pred, average='macro'))


