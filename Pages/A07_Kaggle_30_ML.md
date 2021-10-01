# Kaggle 30 Days of Machine Learning

# Table of contents
- [1. Data Pre-Processing](#1-data-pre-processing)
  - [1.1. Read and Split Data](#11-read-and-split-data) 
  - [1.2. Missing Values](#12-missing-values)
  - [1.3. Categorical variable](#13-categorical-variable)
  - [1.4. Pipelines](#14-pipelines)
- [2. EDA](#2-eda)
  - [2.1. Graph](#21-graph) 
- [3. Feature Engineering](#3-feature-engineering)
- [4. Model Training](#4-model-training)
  - [4.1. Underfitting and Overfitting](#41-underfitting-and-overfitting)
  - [4.2. Evaluation Metrics](#42-evaluation-metrics)
    - [4.2.1. Metrics for Regression](#421-mectrics-for-regression)
- [5. Ensemble methods](#5-ensemble-methods)
  - [5.1. Random Forests](#51-random-forests)
  - [5.2. Gradient Boosting](#52-gradient-boosting) 

# 1. Data Pre-Processing
## 1.1. Read and Split Data
- **Step 1**: Read the data
```Python
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Obtain target and predictors
y = X_full["target"]

X = X_full[:-1].copy() #X will not include last column, which is "target" column
X_test = X_test_full.copy()
```
- **Step 2**: Break off validation set from training data `X`
```Python
# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
```
- **Step 3**: Comparing different models
```Python
models = [model_1, model_2, model_3, model_4, model_5]

# Function for comparing different models
def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

for i in range(0, len(models)):
    mae = score_model(models[i])
    print("Model %d MAE: %d" % (i+1, mae))
```
[(Back to top)](#table-of-contents)

## 1.2. Missing Values
```Python
# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

# Number of missing values in each column of training data
missing_val_count_by_column = (X_train.isnull().sum())
```
- **Method 1**: Drop Columns with Missing Values 
- **Method 2**: Imputation
- **Method 3**: Extension To Imputation

### 1.2.1. Method 1: Drop Columns with Missing Values
<img width="889" alt="Screenshot 2021-08-20 at 10 53 33" src="https://user-images.githubusercontent.com/64508435/130171794-186b7922-3464-4057-9004-87111c6ea44f.png">

```Python
# Drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)
```
### 1.2.2. Method 2: Imputation
- `Imputation` fills in the missing values with some number.
- `strategy = “mean”, "median"` for numerical column
- `strategy = “most_frequent”` for object (categorical) column
<img width="889" alt="Screenshot 2021-08-20 at 10 56 11" src="https://user-images.githubusercontent.com/64508435/130172082-479fbb77-03f9-4438-b8bc-97bcbe3e0d1e.png">

```Python
from sklearn.impute import SimpleImputer

# Imputation
my_imputer = SimpleImputer(missing_values=np.nan, strategy="mean") 
#Only fit on training data
my_imputer.fit(X_train) 

imputed_X_train = pd.DataFrame(my_imputer.transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Fill in the lines below: imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns
```
### 1.2.2. Method 3: Extension To Imputation
- Imputation is the standard approach, and it usually works well. 
- However, imputed values may be systematically above or below their actual values (which weren't collected in the dataset). Or rows with missing values may be unique in some other way. 
- In that case, your model would make better predictions by considering which values were originally missing.

<img width="889" alt="Screenshot 2021-08-20 at 11 36 52" src="https://user-images.githubusercontent.com/64508435/130175336-a19e86d8-cba1-489a-87cb-c33c9378f8c0.png">

- **Note**: In some cases, this will meaningfully improve results. In other cases, it doesn't help at all.
```Python
# Make copy to avoid changing original data (when imputing)
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

# Make new columns indicating what will be imputed
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

# Imputation removed column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns
```

[(Back to top)](#table-of-contents)

## 1.3. Categorical variable
- There are 4 types of Categorical variable
  - `Nominal`: non-order variables like "Honda", "Toyota", and "Ford"
  - `Ordinal`: the order is important 
    - For tree-based models (like decision trees and random forests), you can expect ordinal encoding to work well with ordinal variables
    - `Label Encoder` &#8594; can map to 1,2,3,4, etc &#8594; Use **Tree-based Models: Random Forest, GBM, XGBoost**
    - `Binary Encoder` &#8594; binary-presentation vectors of 1,2,3,4, etc values &#8594; Use **Logistic and Linear Regression, SVM**
  - `Binary`: only have 2 values (Female, Male)
  - `Cyclic`: Monday, Tuesday, Wednesday, Thursday
- Determine Categorical Columns:
```Python
# Categorical columns in the training data 
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]
```
- **Filter Good & Problematic Categorical Columns** which will affect Encoding Procedure:
  - For example: Unique values in Train Data are different from Unique values in Valid Data &#8594; Solution: ensure values in `Valid Data` is a subset of values in `Train Data`
  - The simplest approach, however, is to drop the problematic categorical columns.
```Python
# Categorical columns in the training data
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

# Columns that can be safely ordinal encoded
good_label_cols = [col for col in object_cols if 
                   set(X_valid[col]).issubset(set(X_train[col]))]
        
# Problematic columns that will be dropped from the dataset
bad_label_cols = list(set(object_cols)-set(good_label_cols))
        
print('Categorical columns that will be ordinal encoded:', good_label_cols)
print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)
```
  - The simplest approach, however, is to drop the problematic categorical columns.
```Python
# Drop categorical columns that will not be encoded
label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)
```
There are 5 methods to encode Categorical variables 
- **Method 1**: Drop Categorical Variables 
- **Method 2**: Ordinal Encoding
- **Method 3**: Label Encoding (Same as Ordinal Encoder but NOT care about the order)
- **Method 4**: One-Hot Encoding
- **Method 5**: Entity Embedding (Need to learn from Video: https://youtu.be/EATAM3BOD_E)

### 1.3.1. Method 1: Drop Categorical Variables 
- This approach will only work well if the columns did not contain useful information.
```Python
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])
```
### 1.3.2. Method 2: Ordinal Encoding
<img width="764" alt="Screenshot 2021-08-22 at 18 00 58" src="https://user-images.githubusercontent.com/64508435/130351069-8cd904d8-f59d-4c6e-a454-1b636c81c2e2.png">

- This approach assumes an ordering of the categories: "Never" (0) < "Rarely" (1) < "Most days" (2) < "Every day" (3).

```Python
from sklearn.preprocessing import OrdinalEncoder

# Apply ordinal encoder 
ordinal_encoder = OrdinalEncoder() # Your code here
ordinal_encoder.fit(label_X_train[good_label_cols])

label_X_train[good_label_cols] = ordinal_encoder.transform(label_X_train[good_label_cols])
label_X_valid[good_label_cols] = ordinal_encoder.transform(label_X_valid[good_label_cols])
```
### 1.3.3. Method 3: Label Encoding
- Same as Ordinal Encoder but NOT care about the order, but follow by Alphabet of the values
- `Label Encoder` need to **fit in each column separately**
```Python
from sklearn.preprocessing import LabelEncoder

# Drop categorical columns that will not be encoded
label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)


for c in good_label_cols:
    label_encoder = LabelEncoder()
    label_encoder.fit(label_X_train[c])
    label_X_train[c] = label_encoder.transform(label_X_train[c])
    label_X_valid[c] = label_encoder.transform(label_X_valid[c])
```
### 1.3.4. Method 4: One-Hot Encoding
#### Investigating Cardinality
- `Cardinality`: # of unique entries of a categorical variable
  - For instance, the `Street` column in the training data has two unique values: `Grvl` and `Pave`, the `Street` col has cardinality 2
- For large datasets with many rows, one-hot encoding can greatly expand the size of the dataset. 
- Hence, we typically will only one-hot encode columns with relatively `low cardinality`. 
- `High cardinality` columns can either be dropped from the dataset, or we can use ordinal encoding.
```Python
# Columns that will be one-hot encoded
low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]

# Columns that will be dropped from the dataset
high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))

print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)
print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)
```
#### One-Hot Encoding
- One-hot encoding generally does NOT perform well if the categorical variable has `cardinality >= 15` as One-Hot encoder will expand the original training data with increasing columns

<img width="764" alt="Screenshot 2021-08-22 at 18 33 33" src="https://user-images.githubusercontent.com/64508435/130351973-e54a71c1-c010-4233-a282-37e5528eaccd.png">

- Set `handle_unknown='ignore'` to avoid errors when the validation data contains classes that aren't represented in the training data, and
- Set `sparse=False` ensures that the encoded columns are returned as a numpy array (instead of a sparse matrix).

```Python
from sklearn.preprocessing import OneHotEncoder

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_encoder.fit(X_train[low_cardinality_cols])
OH_cols_train = pd.DataFrame(OH_encoder.transform(X_train[low_cardinality_cols])) #Convert back to Pandas DataFrame from Numpy Array
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))  

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns in the original datasets (will replace with one-hot encoding columns)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
```

## 1.4. Pipelines
- **Pipelines** are a simple way to keep your data preprocessing and modeling code organized.

[(Back to top)](#table-of-contents)

# 2. EDA
## 2.1. Graph
### 2.1.1. Label
#### Overwrite Label:
```Python
# Get current yticks: An array of the values displayed on the y-axis (150, 175, 200, etc.)
ticks = ax.get_yticks()
# Format those values into strings beginning with dollar sign
new_labels = [f"${int(tick)}" for tick in ticks]
# Set the new labels
ax.set_yticklabels(new_labels)
```
[(Back to top)](#table-of-contents)

# 3. Feature Engineering

# 4. Model Training
## 4.1. Underfitting and Overfitting
Models can suffer from either:
- **Overfitting**: capturing spurious patterns that won't recur in the future, leading to less accurate predictions
  - Where a model matches the training data almost perfectly, but does poorly in validation and other new data.  
- **Underfitting**: failing to capture relevant patterns, again leading to less accurate predictions.
  - When a model fails to capture important distinctions and patterns in the data, so it performs poorly even in training data 
### 4.1.1. Methods to avoid Underfitting and Overfitting
#### Example 1: DecisionTreeRegressor Model
- `max_leaf_nodes` argument provides a very sensible way to control overfitting vs underfitting. The more leaves we allow the model to make, the more we move from the underfitting area in the above graph to the overfitting area.
<p align="center"><img src="https://user-images.githubusercontent.com/64508435/129434680-b30efd3e-ab04-4871-85ce-03b53027c0e7.png" height="320px" /></p>

- We can use a utility function to help compare MAE scores from different values for `max_leaf_nodes`:
```Python
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
```
- Call the *get_mae* function on each value of max_leaf_nodes. Store the output in some way that allows you to select the value of `max_leaf_nodes` that gives the most accurate model on your data.
```Python
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(scores.keys(), key=(lambda k: scores[k]))
```
[(Back to top)](#table-of-contents)

## 4.2. Evaluation Metrics
- Evaluation Metric used for Competition usually will be specified in Kaggle Competition > Evaluation 
<img width="951" alt="Screenshot 2021-08-15 at 16 25 45" src="https://user-images.githubusercontent.com/64508435/129472164-4101cc49-0320-4094-a9c4-8a5f697e30b6.png">


### 4.2.1. Metrics for Regression
#### Mean Absolute Error (MAE)
```Python
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_pred, y_test)
```
#### Root Mean Squared Error (RMSE)
```Python
import numpy as np
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(y_pred, y_test))
```


# 5. Ensemble methods
- The goal of `ensemble methods` is to combine the predictions of several base estimators built with a given learning algorithm in order to improve generalizability / robustness over a single estimator (**for classification, regression and anomaly detection**)
- Two families of ensemble methods:
  - In **averaging methods**, the driving principle is to build several estimators independently and then to average their predictions. On average, the combined estimator is usually better than any of the single base estimator because its variance is reduced.
    - *Examples*: Bagging methods, Forests of randomized trees, etc.
  - In **boosting methods**, base estimators are built sequentially and one tries to reduce the bias of the combined estimator. The motivation is to combine several weak models to produce a powerful ensemble.
    - *Examples*: AdaBoost, Gradient Tree Boosting, etc.

## 5.1. Random Forests
- Decision trees leave you with a difficult decision. 
  - A deep tree with lots of leaves will overfit because each prediction is coming from historical data from only the few data at its leaf. 
  - But a shallow tree with few leaves will perform poorly because it fails to capture as many distinctions in the raw data.
- The random forest uses many trees, and it makes a prediction by averaging the predictions of each component tree. 
- It generally has much better predictive accuracy than a single decision tree and it works well with default parameters.

## 5.2. Gradient Boosting
```Python
#Example of Gradient Boosting - Regressor
from sklearn.ensemble import GradientBoostingRegressor

gbm_model = GradientBoostingRegressor(random_state=1, n_estimators=500)
gbm_model.fit(train_X, train_y)
gbm_val_predictions = gbm_model.predict(val_X)
gbm_val_rmse = np.sqrt(mean_squared_error(gbm_val_predictions, val_y))
```

# Submission
```Python
predictions = model.predict(X_test)
output = pd.DataFrame({'id': test_data.id, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
```


[(Back to top)](#table-of-contents)
