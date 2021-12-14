# Kaggle 30 Days of Machine Learning

# Table of contents
- [1. Data Pre-Processing](#1-data-pre-processing)
  - [1.1. Read and Split Data](#11-read-and-split-data) 
  - [1.2. Missing Values](#12-missing-values)
  - [1.3. Categorical variable](#13-categorical-variable)
  - [1.4. Pipelines](#14-pipelines)
  - [1.5. Cross-Validation](#15-cross-validation)
- [2. EDA](#2-eda)
  - [2.1. Graph](#21-graph) 
- [3. Feature Engineering](#3-feature-engineering)
- [4. Model Training](#4-model-training)
  - [4.1. Underfitting and Overfitting](#41-underfitting-and-overfitting)
  - [4.2. Evaluation Metrics](#42-evaluation-metrics)
    - [4.2.1. Metrics for Regression](#421-metrics-for-regression)
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
- **Step 2**: Investigate and filter Numeric & Categorical Data
  - Note 1: Some features although they are numerical, but there data type is object, and vice versa. Hence, **need to spend time to investigate on the real type of the features**, *convert them into correct data type before performing the below commands*.
  ```Python
  # Select numeric columns only
  numeric_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
  # Categorical columns in the training data
  object_cols = [col for col in X.columns if X[col].dtype == "object"]
  ```
- **Step 3**: Break off validation set from training data `X`
```Python
# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
```
- **Step 4**: Comparing different models
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
- This part is to handle missing values for both `Numerical` & `Categorical` Data
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
  - If a categorical attribute has a large number of possible categories, then one-hot encoding will result in a large number of input features. This may slow down training and degrade performance.
  - If this happens, you will want to produce denser representations called `embeddings`, but this requires a good understanding of neural networks (see [Chapter 14](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781491962282/ch14.html#rnn_chapter) for more details).
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
- Specifically, a pipeline bundles preprocessing and modeling steps so you can use the whole bundle as if it were a single step.
- Construct the full pipeline in three steps:
  - **Step 1: Define Preprocessing Steps**
    - A pipeline bundles together preprocessing and modeling steps, we use the `ColumnTransformer` class to bundle together different preprocessing steps. 
      - imputes missing values in numerical data, and
      - imputes missing values and applies a one-hot encoding to categorical data.
    ```Python
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder

    # Preprocessing for numerical data
    numerical_transformer = SimpleImputer(strategy='constant')

    # Preprocessing for categorical data using Pipeline class
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data using ColumnTransformer class
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    ```
  - **Step 2: Define the Model**
    ```Python
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import GradientBoostingRegressor

    model1 = RandomForestRegressor(n_estimators=100, random_state=0)
    model2 = GradientBoostingRegressor(n_estimators=500, random_state = 42)
    ```
  - **Step 3: Create and Evaluate the Pipeline**
    - Finally, we use the Pipeline class to define a pipeline that bundles the preprocessing and modeling steps. There are a few important things to notice:
      - With the pipeline, we preprocess the training data and fit the model in a single line of code. *(In contrast, without a pipeline, we have to do imputation, one-hot encoding, and model training in separate steps. This becomes especially messy if we have to deal with both numerical and categorical variables!)*
      - With the pipeline, we supply the unprocessed features in X_valid to the predict() command, and the pipeline automatically preprocesses the features before generating predictions. *(However, without a pipeline, we have to remember to preprocess the validation data before making predictions.)*
    ```Python
    from sklearn.ensemble import GradientBoostingRegressor
    my_pipeline1 = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('rf', model1)
                                 ])
    my_pipeline2 = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('gbm', model2)
                                 ])

    # Preprocessing of training data, fit model 
    my_pipeline1.fit(X_train, y_train)
    # Preprocessing of validation data, get predictions
    preds1 = my_pipeline1.predict(X_valid)

    my_pipeline2.fit(X_train, y_train)
    preds2 = my_pipeline2.predict(X_valid)

    preds = (preds1 + preds2)/2
    # Evaluate the model
    score = mean_absolute_error(y_valid, preds)
    print('MAE:', score)
    ```
[(Back to top)](#table-of-contents)


## 1.5. Cross-Validation
- **For small dataset &#8594; Cross-validation**, we run our modeling process on different subsets of the data to get multiple measures of model quality.
  - **Stratified k-fold**: Stratified k-fold cross-validation is same as just k-fold cross-validation, but in Stratified k-fold cross-validation, it does stratified sampling instead of random sampling.
  - For example, the US population is composed of 51.3% female and 48.7% male, so a well-conducted survey in the US would try to maintain this ratio in the sample: 513 female and 487 male. This is called `stratified sampling`: the population is divided into homogeneous subgroups called `strata`, and the right number of instances is sampled from each stratum to guarantee that the test set is representative of the overall population. If they used purely random sampling, there would be about 12% chance of sampling a skewed test set with either less than 49% female or more than 54% female. Either way, the survey results would be significantly biased.
    - Hence, Stratified k-fold keeps the same ratio of classes in each fold in comparison with the ratio of the original training data.
    - **Classification** problem: can apply Stratified k-fold directly
    - **Regression** problem: need to convert `Y` into `1+log2(N)` bins (Sturge’s Rule) and then Stratified k-fold  will split accordingly.
  ![image](https://user-images.githubusercontent.com/64508435/144378824-53f0db43-38f1-47cf-a0c2-15bf74f9d2ab.png)

  ```Python
  from sklearn.model_selection import cross_val_score
  
  def get_score(n_estimators):
      my_pipeline = Pipeline(steps=[
          ('preprocessor', SimpleImputer()),
          ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=0))
      ])
      
      # Multiply by -1 since sklearn calculates *negative* MAE
      scores = -1 * cross_val_score(my_pipeline, X, y,
                                cv=5, #This is 5-fold cross-validation
                                scoring='neg_mean_absolute_error')
      #Since cross_val_score return 5 MAE for each fold, so take mean()
      return scores.mean() 
  
  #Evaluate the model performance corresponding to eight different values for the number of trees (n_estimators) in the random forest: 50, 100, 150, ..., 300, 350, 400.
  results = {n_estimators: get_score(n_estimators) for n_estimators in range(50, 450, 50)} 
  
  plt.plot(results.keys(), results.values())
  plt.show()
  ```
  <img width="414" alt="Screenshot 2021-12-02 at 16 42 21" src="https://user-images.githubusercontent.com/64508435/144396996-81ae36c5-98c1-4a50-b9df-dee77eaf44cd.png">

- **For large dataset &#8594; Hold-out**: when `training data > 100K or 1M`, we will hold-out 5-10% data as a validation set.

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
- Random forest method, which achieves better performance than a single decision tree simply by averaging the predictions of many decision trees. We refer to the random forest method as an "ensemble method". By definition, ensemble methods combine the predictions of several models (e.g., several trees, in the case of random forests).
- Decision trees leave you with a difficult decision. 
  - A deep tree with lots of leaves will overfit because each prediction is coming from historical data from only the few data at its leaf. 
  - But a shallow tree with few leaves will perform poorly because it fails to capture as many distinctions in the raw data.
- The random forest uses many trees, and it makes a prediction by averaging the predictions of each component tree. 
- It generally has much better predictive accuracy than a single decision tree and it works well with default parameters.

## 5.2. Gradient Boosting
- **Gradient boosting** is the method dominates many Kaggle competitions and achieves state-of-the-art results on a variety of datasets.
- Gradient boosting is a method that goes through cycles to iteratively add models into an ensemble.
- It begins by initializing the ensemble with a single model, whose predictions can be pretty naive. (Even if its predictions are wildly inaccurate, subsequent additions to the ensemble will address those errors.)
- *"Gradient"* in "gradient boosting" refers to the fact that we'll use `gradient descent` on the loss function to determine the parameters in this new model.)
![image](https://user-images.githubusercontent.com/64508435/144753278-7e6573ec-4a2e-45e9-aa2b-b713529366d0.png)


```Python
#Example of Gradient Boosting - Regressor
from sklearn.ensemble import GradientBoostingRegressor

gbm_model = GradientBoostingRegressor(random_state=1, n_estimators=500)
gbm_model.fit(train_X, train_y)
gbm_val_predictions = gbm_model.predict(val_X)
gbm_val_rmse = np.sqrt(mean_squared_error(gbm_val_predictions, val_y))
```

### 5.2.1. XGBoost 
- `XGBoost` stands for **extreme gradient boosting**, which is an implementation of gradient boosting with several additional features focused on performance and speed.
- XGBoost has a few parameters that can dramatically affect accuracy and training speed:
  - `n_estimators`: (typically range from **100-1000**, though this depends a lot on the learning_rate parameter) specifies how many times to go through the modeling cycle described above. It is equal to the number of models that we include in the ensemble.
    - Too low a value causes *underfitting*, which leads to inaccurate predictions on both training data and test data.
    - Too high a value causes *overfitting*, which causes accurate predictions on training data, but inaccurate predictions on test data (which is what we care about).
  - `learning_rate`: (default: learning_rate=0.1) Instead of getting predictions by simply adding up the predictions from each component model, we can multiply the predictions from each model by a small number (known as the **learning rate**) before adding them in.
  - `n_jobs`: equal to the number of cores on your machine
    - On smaller datasets, the resulting model won't be any better, so micro-optimizing for fitting time is typically nothing but a distraction.
    - On larger datasets where runtime is a consideration, you can use parallelism to build your models faster; otherwise spend a long time waiting during the `fit` command.
```Python
from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4, random_state = 42)

my_model.fit(X_train, y_train, 
             #Setting early_stopping_rounds=5 is a reasonable choice. In this case, we stop after 5 straight rounds of deteriorating validation scores
             early_stopping_rounds=5, 
             #When using early_stopping_rounds, you also need to set aside some data for calculating the validation scores - this is done by setting the eval_set parameter.
             eval_set=[(X_valid, y_valid)], 
             verbose=False)
```
  - `early_stopping_rounds`: offers a way to automatically find the ideal value for n_estimators. Early stopping causes the model to stop iterating when the validation score stops improving, even if we aren't at the hard stop for n_estimators. It's smart to set a high value for n_estimators and then use early_stopping_rounds to find the optimal time to stop iterating.
    - Setting **early_stopping_rounds=5** is a reasonable choice. In this case, we stop after 5 straight rounds of deteriorating validation scores.

  

# Submission
```Python
predictions = model.predict(X_test)
output = pd.DataFrame({'id': test_data.id, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
```


[(Back to top)](#table-of-contents)
