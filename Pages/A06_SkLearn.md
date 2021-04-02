# Scikit-Learn
# Table of contents
- [Table of contents](#table-of-contents)
- [Scikit-Learn Introduction](#scikit-learn-introduction)
- [Scikit-Learn Workflow](#scikit-learn-workflow)
  - [1. Get data ready](#get-data-ready)

# Scikit Learn Introduction
- Scikit Learn (SkLearn): Python Machine Learning Library, built on Numpy & Matplotlib
- Machine Learning = Computer is writting it own function (or ML Models/Algorithms) based on I/P & O/P data.

[(Back to top)](#table-of-contents)

# Scikit Learn Workflow
## Get data ready
4 main things we have to do:
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
