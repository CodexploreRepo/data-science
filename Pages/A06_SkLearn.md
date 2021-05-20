# Scikit-Learn
# Table of contents
- [Table of contents](#table-of-contents)
- [Scikit-Learn Introduction](#scikit-learn-introduction)
- [Scikit-Learn Workflow](#scikit-learn-workflow)
  - [1. Get data ready](#get-data-ready)
  - [2. Choose the right estimator](#choose-the-right-estimator)
  - [3. Make predictions using ML model](#make-predictions-using-ml-model)
  - [4. Evaluate a Machine Learning Model](#evaluate-a-machine-learning-model)

# Scikit Learn Introduction
- Scikit Learn (SkLearn): Python Machine Learning Library, built on Numpy & Matplotlib
- Machine Learning = Computer is writting it own function (or ML Models/Algorithms) based on I/P & O/P data.

[(Back to top)](#table-of-contents)

# Scikit Learn Workflow
## Get data ready
### 4 main things we have to do:
1. Split the data into features and labels (Usually `X` and `y`)
2. Imputing: Filling or disregarding missing values
3. Feature Encoding: Converting non-numerical values to numerical values
4. Feature Scaling: making sure all of your numerical data is on the same scale

#### 1. Split Data into X and y
- Before split, Drop all rows with Missing Values in y.
```Python
# Drop the rows with missing in the "Price" column
car_sales_missing.dropna(subset=["Price"], inplace=True)
```
- Split Data into X and y
```Python
# Create X (features matrix)
X = car_sales.drop("Price", axis = 1) # Remove 'Price' column (y)

# Create y (lables)
y = car_sales["Price"]
```
- Split X, y into Training & Test Sets

```Python
np.random.seed(42)

# Split the data into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
```
#### 2. Imputing
- Fill missing values with Scikit-Learn `SimpleImputer()` transforms data by filling missing values with a given strategy 
```Python
from sklearn.impute import SimpleImputer #Help fill the missing values
from sklearn.compose import ColumnTransformer

# Fill Categorical values with 'missing' & numerical values with mean
cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")
num_imputer = SimpleImputer(strategy="mean")

# Define different column features
categorical_features = ["Make", "Colour"]
numerical_feature = ["Odometer (KM)"]

imputer = ColumnTransformer([
    ("cat_imputer", cat_imputer, categorical_features),
    ("num_imputer", num_imputer, numerical_feature)])
```
**Note:** We use fit_transform() on the training data and transform() on the testing data. 
* In essence, we learn the patterns in the training set and transform it via imputation (fit, then transform). 
* Then we take those same patterns and fill the test set (transform only).

```Python
# learn the patterns in the training set and transform it via imputation (fit, then transform)
filled_X_train = imputer.fit_transform(X_train)
# take those same patterns and fill the test set (transform only)
filled_X_test = imputer.transform(X_test)
```
- Convert back the filled columns back to Data Frame
```Python
# Get our transformed data array's back into DataFrame's
car_sales_filled_train = pd.DataFrame(filled_X_train, 
                                      columns=["Make", "Colour", "Odometer (KM)"])

car_sales_filled_test = pd.DataFrame(filled_X_test, 
                                      columns=["Make", "Colour", "Odometer (KM)"])
```

#### 3. Feature Encoding: Converting categorical features into numerical values
- Note: **Needs to inspect numerical features to check their data are categorical or not** &#8594; need to convert into categorical also.
- For example: "Door" feature, although, is numerical in type, but actually categorical feature since only  3 options: (4,5,3)
```Python
# Inspect whether "Door" is categorical feature or not
# Although "Door" contains numerical values
car_sales["Doors"].value_counts()

# Conclusion: "Door" is categorical feature since it has only 3 options: (4,5,3)
4    856
5     79
3     65
Name: Doors, dtype: int64
```

```Python
# Turn the categories into numbers
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_features = ["Make", "Colour", "Doors"] 

one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot", 
                                  one_hot,
                                  categorical_features)], remainder="passthrough")

# Fill train and test values separately
transformed_X_train = transformer.fit_transform(car_sales_filled_train)
transformed_X_test = transformer.transform(car_sales_filled_test)

transformed_X_train.toarray()
```
#### 4. Feature Scaling
- For example: predict the sale price of cars 
  - The number of kilometres on their odometers varies from 6,000 to 345,000 
  - The median previous repair cost varies from 100 to 1,700. 
  - A machine learning algorithm may have trouble finding patterns in these wide-ranging variables
- To fix this, there are two main types of feature scaling:
  -  **Normalization** (also called `min-max scaling`): This rescales all the numerical values to between 0 and 1 &#8594; `MinMaxScaler` from Scikit-Learn.
  - **Standardization**: This subtracts the mean value from all of the features (so the resulting features have 0 mean). It then scales the features to unit variance (by dividing the feature by the standard deviation). &#8594;  `StandardScalar` class from Scikit-Learn.
- Note: 
  - Feature scaling usually isn't required for your target variable + encoded feature variables
  - Feature scaling is usually not required with tree-based models (e.g. Random Forest) since they can handle varying features

##### Readings
* [Feature Scaling- Why it is required?](https://medium.com/@rahul77349/feature-scaling-why-it-is-required-8a93df1af310)
* [Feature Scaling with scikit-learn](https://benalexkeen.com/feature-scaling-with-scikit-learn/)
* [Feature Scaling for Machine Learning: Understanding the Difference Between Normalization vs. Standardization](https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/)

[(Back to top)](#table-of-contents)

## Choose the right estimator
- Scikit-learn uses **estimator** as another term for machine learning model or algorithm
- Based on the .score() + ML Map to choose right estimator
- Map: https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
1. **Structured data (tables)** → ensemble methods (combine the predictions of several base estimators built with a given learning algorithm in order to improve generalizability / robustness over a single estimator)
2. **Unstructured data (image, audio, text, video)** → deep learning or transfer learning 

### Choose the right estimator for Regression Problem:
```Python
# Let's try the Ridge Regression Model
from sklearn.linear_model import Ridge

#Setup random seed
np.random.seed(42) #to make sure result is reproducible

#instantiate Ridge Model
model = Ridge()
model.fit(X_train, y_train)

# Check the score of the Ridge model on test data
model.score(X_test, y_test) #Return R^2 of the regression
```

### Choose the right estimator for Classification Problem:

```Python
# Import the LinearSVC estimator class
from sklearn.ensemble import RandomForestClassifier 

# Setup random seed
np.random.seed(42)

# Instantiate Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100)

#Fit the model to the data (training the machine learning model)
clf.fit(X_train, y_train)

# Evaluate Random Forest Classifier (use the parterns the model has learnt)
clf.score(X_test, y_test) #Return the mean accuracy on the given test data and labels.
```

[(Back to top)](#table-of-contents)


## Make predictions using ML model
### 3.1 Predict for Classification Models
#### 2 ways to make predictions:
1. Using `predict()`
```Python
# Use a trained model to make predictions
y_preds = clf.predict(X_test)
```
**Predict a single value**: "predict" method always expects a 2D array as the format of its inputs. And putting 12 into a double pair of square brackets makes the input exactly a 2D array:
```Python
clf.predict([[12]])
```
2. Using `predict_proba()`
- `predict_proba()` returns the probabilities of a classification label.
```Python
clf.predict_proba(X_test) #[x% prob class = 0, y% prob class = 1]

array([[0.89, 0.11],
       [0.49, 0.51],
       [0.43, 0.57],
       [0.84, 0.16],
       [0.18, 0.82]])
```
- This output `[0.89, 0.11]` means the model is predicting label 0 (index 0) with a probability score of 0.89.
- Because the score is over 0.5, when using predict(), a label of 0 is assigned.

### 3.2 Predict for Regression Models
- `predict()` can also be used for regression models

[(Back to top)](#table-of-contents)

## Evaluate a Machine Learning Model
* Tips: Google 'scikit learn evaluate a model'
* 3 ways to evaluate Scikit Learn Models
1. Estimator `score` method
2. The `scoring` parameter
3. Problem-specific metric function

### 4.1 Evaluate a model with `Score` Method
* Note: Calling `score()` on a model instance will return a metric assosciated with the type of model you're using. The metric depends on which model you're using.
- Regression Model: `model.score(X_test, y_test) #score() = Return R^2 of the regression`
- Classifier Model: `clf.score(X_test, y_test)   #score() = Return the mean accuracy on the given test data and labels.`

### 4.2 Evaluating a model using the `scoring` parameter
* This parameter can be passed to methods such as `cross_val_score()` or `GridSearchCV()` to tell Scikit-Learn to use a specific type of scoring metric.
* `cross_val_score()` vs `score()`
```Python
from sklearn.model_selection import cross_val_score
cross_val_score(clf, X, y, cv = 10) #by default = 10-fold => split X,y into 10 different dataset and train 5 different models

array([0.90322581, 0.80645161, 0.87096774, 0.9       , 0.86666667,
       0.76666667, 0.7       , 0.83333333, 0.73333333, 0.8       ])
```
* `cross_val_score()` returns an array where as `score()` only returns a single number

```Python
# Using score()
clf.score(X_test, y_test)
```
* Figure 1.0: using score(X_test, y_test), a model is trained using the training data or 80% of samples, this means 20% of samples aren't used for the model to learn anything
* Figure 2.0: using 5-fold cross-validation, instead of training only on 1 training split and evaluating on 1 testing split, 5-fold cross-validation does it 5 times. On a different split each time, returning a score for each

![image](https://user-images.githubusercontent.com/64508435/118992921-a6970180-b9b7-11eb-9592-2207dfad8e09.png)

* Note#1: if you were asked to report the accuracy of your model, even though it's lower, you'd prefer the cross-validated metric over the non-cross-validated metric.
* Note#2:`cross_val_score(clf, X, y, cv=5, scoring=None) # default scoring`: by default, scoring set to `None`, i.e: `cross_val_score()` will use  the same metric as score()
  - For Ex: clf which is an instance of RandomForestClassifier uses mean accuracy as the default score() metric, so `cross_val_score()` will use mean accuracy also
  - You can change the **evaluation score** of `cross_val_score()` uses by changing the `scoring` parameter.


[(Back to top)](#table-of-contents)
