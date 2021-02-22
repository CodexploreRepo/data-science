# Regression
# Table of contents

- [Table of contents](#table-of-contents)
- [Introduction to Regressions](#introduction-to-regressions)
- [Simple Linear Regression](#simple-linear-regression)
  - [Outline: Building a Model](#outline-building-a-model)
  - [Creating a Model](#creating-a-model)
  - [Predicting a Test Result](#predicting-a-test-result)
  - [Visualising the Test set results](#visualising-the-test-set-results)
  - [Getting Linear Regression Equation](#getting-linear-regression-equation)
  - [Evaluating the Algorithm](#evaluating-the-algorithm)
    - [R Square or Adjusted R Square](#r-square-or-adjusted-r-square)
    - [Mean Square Error (MSE)/Root Mean Square Error (RMSE)](#mean-square-error-and-root-mean-square-error)
    - [Mean Absolute Error (MAE)](#mean-absolute-error)
- [Multiple Linear Regression](#multiple-linear-regression) 
  - [Assumptions of Linear Regression](#assumptions-of-linear-regression)
  - [Dummy Variables](#dummy-variables)
  - [Understanding P-value](#understanding-p-value)
  - [Building a Model](#building-a-model)
- [Polynomial Linear Regression](#polynomial-linear-regression) 


# Introduction to Regressions
- Simple Linear Regression    : `y = b0 + b1*x1`
- Multiple Linear Regression  : `y = b0 + b1*x1 + b2*x2 + ... + bn*xn`
- Polynomial Linear Regression: `y = b0 + b1*x1 + b2*x1^(2) + ... +  bn*x1^(n)`

# Simple Linear Regression
## Outline Building a Model
- Importing libraries and datasets
- Splitting the dataset
- Training the simple Linear Regression model on the Training set
- Predicting and visualizing the test set results
- Visualizing the training set results 
- Making a single prediction 
- Getting the final linear regression equation (with values of the coefficients) 
```
y = bo + b1 * x1
```
- y: Dependent Variable (DV)
- x: InDependent Variable (IV)
- b0: Intercept Coefficient
- b1: Slope of Line Coefficient
<p align="center"><img src="https://user-images.githubusercontent.com/64508435/105186896-24174600-5b6d-11eb-9eb8-8c7f5e82d268.png" height="200" /></p>


## Creating a Model
- Using `sklearn.linear_model`,  `LinearRegression` model
```Python
from sklearn.linear_model import LinearRegression

#To Create Instance of Simple Linear Regression Model
regressor = LinearRegression()

#To fit the X_train and y_train
regressor.fit(X_train, y_train)
```
## Predicting a Test Result
```Python
y_pred = regressor.predict(X_test)
```
### Predict a single value
**Important note:**  "predict" method always expects a 2D array as the format of its inputs. 
- And putting 12 into a double pair of square brackets makes the input exactly a 2D array:
- `regressor.predict([[12]])`

```Python
print(f"Predicted Salary of Employee with 12 years of EXP: {regressor.predict([[12]])}" )

#Output: Predicted Salary of Employee with 12 years of EXP: [137605.23485427]
```
## Visualising the Test set results
```Python
#Plot predicted values
plt.scatter(X_test, y_test, color = 'red', label = 'Predicted Value')
#Plot the regression line
plt.plot(X_train, regressor.predict(X_train), color = 'blue', label = 'Linear Regression')
#Label the Plot
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
#Show the plot
plt.show()
```
![download](https://user-images.githubusercontent.com/64508435/105365689-7c1b7e80-5c39-11eb-8e44-12866fb7eb3d.png)

## Getting Linear Regression Equation
- General Formula: `y_pred = model.intercept_ + model.coef_ * x`
```Python
print(f"b0 : {regressor.intercept_}")
print(f"b1 : {regressor.coef_}")

b0 : 25609.89799835482
b1 : [9332.94473799]
```

Linear Regression Equation: `Salary = 25609 + 9332.94×YearsExperience`

## Evaluating the Algorithm
- compare how well different algorithms perform on a particular dataset.
- For regression algorithms, three evaluation metrics are commonly used:
  1. R Square/Adjusted R Square > Percentage of the output variability
  2. Mean Square Error(MSE)/Root Mean Square Error(RMSE) > to compare performance between different regression models
  3. Mean Absolute Error(MAE) > to compare performance between different regression models
  
### R Square or Adjusted R Square
#### R Square: Coefficient of determination
- R Square measures how much of **variability** in predicted variable can be explained by the model.
- `Variance` is a measure in statistics defined as the average of the square of differences between individual point and the expected value.
- R Square value: between 0 to 1 and bigger value indicates a better fit between prediction and actual value.
- However, it does **not take into consideration of overfitting problem**. 
  - If your regression model has many independent variables, because the model is too complicated, it may fit very well to the training data 
  - but performs badly for testing data.
  - Solution: Adjusted R Square
  
<p align="center"><img src="https://user-images.githubusercontent.com/64508435/105422836-62048f00-5c7f-11eb-99f6-d94b3ebf1784.png" height="60" /></p>

#### Adjusted R Square
- is introduced Since R-square can be increased by adding more number of variable and may lead to the **over-fitting** of the model
- Will penalise additional independent variables added to the model and adjust the metric to **prevent overfitting issue**.

#### Calculate R Square and Adjusted R Square using Python
- In Python, you can calculate R Square using `Statsmodel` or `Sklearn` Package
```Python
import statsmodels.api as sm

X_addC = sm.add_constant(X)

result = sm.OLS(Y, X_addC).fit()

print(result.rsquared, result.rsquared_adj)
# 0.79180307318 0.790545085707

```
- around 79% of dependent variability can be explain by the model and adjusted R Square is roughly the same as R Square meaning the model is quite robust

### Mean Square Error and Root Mean Square Error
- While **R Square** is a **relative measure** of how well the model fits dependent variables
- **Mean Square Error (MSE)** is an **absolute measure** of the goodness for the fit.
- **Root Mean Square Error(RMSE)** is the square root of MSE. 
  - It is used more commonly than MSE because firstly sometimes MSE value can be too big to compare easily.
  - Secondly, MSE is calculated by the square of error, and thus square root brings it back to the same level of prediction error and make it easier for interpretation.

<p align="center"><img src="https://user-images.githubusercontent.com/64508435/105425248-f113a600-5c83-11eb-8d86-ea14b2795d79.png" height="60" /></p>

```Python
from sklearn.metrics import mean_squared_error
import math
print(mean_squared_error(Y_test, Y_predicted))
print(math.sqrt(mean_squared_error(Y_test, Y_predicted)))
# MSE: 2017904593.23
# RMSE: 44921.092965684235
```
### Mean Absolute Error
- Compare to MSE or RMSE, MAE is a more direct representation of sum of error terms.

<p align="center"><img src="https://user-images.githubusercontent.com/64508435/105425768-f2919e00-5c84-11eb-83c4-4b53f903fbbc.png" height="60" /></p>

```Python
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(Y_test, Y_predicted))
#MAE: 26745.1109986
```

[(Back to top)](#table-of-contents)

# Multiple Linear Regression
### Assumptions of Linear Regression:
Before choosing Linear Regression, need to consider below assumptions
1. Linearity
2. Homoscedasticity
3. Multivariate normality
4. Independence of errors
5. Lack of multicollinearity

## Dummy Variables
- Since `State` is categorical variable => we need to convert it into `dummy variable`
- No need to include all dummy variable to our Regression Model => **Only omit one dummy variable**
  - Why ? `dummy variable trap` 
![Screenshot 2021-01-28 at 9 22 08 PM](https://user-images.githubusercontent.com/64508435/106144509-3e75a300-61af-11eb-8240-53bed739b2a1.png)

## Understanding P value
- Ho : `Null Hypothesis (Universe)`
- H1 : `Alternative Hypothesis (Universe)`
- For example:
  - Assume `Null Hypothesis` is true (or we are living in Null Universe)

![Screenshot 2021-01-28 at 9 45 01 PM](https://user-images.githubusercontent.com/64508435/106148653-4421b780-61b4-11eb-91b4-4db1247a1a2a.png)

[(Back to top)](#table-of-contents)
## Building a Model
- 5 methods of Building Models
### Method 1: All-in
- Throw in all variables in the dataset
- Usage:
  - Prior knowledge about this problem; OR
  - You have to (Company Framework required)
  - Prepare for Backward Elimination 
### Method 2 [Stepwise Regression]: Backward Elimination (Fastest)
- Step 1: Select a significance level (SL) to stay in the model (e.g: SL = 0.05)
```Python
# Building the optimal model using Backward Elimination
import statsmodels.api as sm

# Avoiding the Dummy Variable Trap by excluding the first column of Dummy Variable
# Note: in general you don't have to remove manually a dummy variable column because Scikit-Learn takes care of it.
X = X[:, 1:]

#Append full column of "1"s to First Column of X using np.append
#Since y = b0*(1) + b1 * x1 + b2 * x2 + .. + bn * xn, b0 is constant and can be re-written as b0 * (1)
#np.append(arr = the array will add to, values = column to be added, axis = row/column)
# np.ones((row, column)).astype(int) => .astype(int) to convert array of 1 into integer type to avoid data type error
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)

#Initialize X_opt with Original X by including all the column from #0 to #5
X_opt = np.array(X[:, [0, 1, 2, 3, 4, 5]], dtype=float) 
#If you are using the google colab to write your code, 
# the datatype of all the features is not set to float hence this step is important: X_opt = np.array(X[:, [0, 1, 2, 3, 4, 5]], dtype=float)
```
- Step 2: Fit the full model with all possible predictors
```Python
#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
```
- Step 3: Consider Predictor with Highest P-value
  - If P > SL, go to Step 4, otherwise go to  [**FIN** : Your Model Is Ready]
- Step 4: Remove the predictor
```Python
#Remove column = 2 from X_opt since Column 2 has Highest P value (0.99) and > SL (0.05).
X_opt = np.array(X[:, [0, 1, 3, 4, 5]], dtype=float) 
#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
```
- Step 5: Re-Fit model without this variable

### Method 3 [Stepwise Regression]: Forward Selection
- Step 1: Select a significance level (SL) to enter in the model (e.g: SL = 0.05)
- Step 2: Fit all simple regression models (y ~ xn). Select the one with Lowest P-value for the independent variable.
- Step 3: Keep this variable and fit all possible regression models with one extra predictor added to the one(s) you already have. 
- Step 4: Consider the predicotr with Lowest P-value. If P < SL (i.e: model is good), go STEP 3 (to add 3rd variable into the model and so on with all variables we have left), otherwise go to  [**FIN** : Keep the previous model]
### Method 4 [Stepwise Regression]: Bidirectional Elemination
- Step 1: Select a significant level to enter and to stay in the model: `e.g: SLENTER = 0.05, SLSTAY = 0.05`
- Step 2: Perform the next step of Forward Selection (new variables must have: P < SLENTER to enter) 
- Step 3: Perform ALL steps of Backward Elimination (old variables must have P < SLSTAY to stay) => Step 2.
- Step 4: No variables can enter and no old variables can exit => [**FIN** : Your Model Is Ready]

### Method 5: Score Comparison
- Step 1: Select a criterion of goodness of ift (e.g Akaike criterion)
- Step 2: Construct all possible regression Models: `2^(N) - 1` total combinations, where N: total number of variables
- Step 3: Select the one with best criterion => [**FIN** : Your Model Is Ready]

### Code Implementation
- Note: Backward Elimination is irrelevant in Python, because the Scikit-Learn library automatically takes care of selecting the statistically significant features when training the model to make accurate predictions.
##### Step 1: Splitting the dataset into the Training set and Test set
```Python
#no need Feature Scaling (FS) for Multi-Regression Model: y = b0 + b1 * x1 + b2 * x2 + b3 * x3, 
# since we have the coefficients (b0, b1, b2, b3) to compensate, so there is no need FS.
from sklearn.model_selection import train_test_split

# NOT have to remove manually a dummy variable column because Scikit-Learn takes care of it.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```

##### Step 2: Training the Multiple Linear Regression model on the Training set
```Python
#LinearRegression will take care "Dummy variable trap" & feature selection
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```

##### Step 3: Predicting the Test set results
```Python
y_pred = regressor.predict(X_test)
```


##### Step 4: Displaying Y_Pred vs Y_test
- Since this is multiple linear regression, so can not visualize by drawing the graph 

```Python
#To display the y_pred vs y_test vectors side by side
np.set_printoptions(precision=2) #To round up value to 2 decimal places

#np.concatenate((tuple of rows/columns you want to concatenate), axis = 0 for rows and 1 for columns)
#y_pred.reshape(len(y_pred),1) : to convert y_pred to column vector by using .reshape()

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))
```
##### Step 5: Getting the final linear regression equation with the values of the coefficients
```Python
print(regressor.coef_)
print(regressor.intercept_)

[ 8.66e+01 -8.73e+02  7.86e+02  7.73e-01  3.29e-02  3.66e-02]
42467.52924853204
```

Equation: 
Profit = 86.6 x DummyState1 - 873 x DummyState2 + 786 x DummyState3 - 0.773 x R&D Spend + 0.0329 x Administration + 0.0366 x Marketing Spend + 42467.53

[(Back to top)](#table-of-contents)


# Polynomial Linear Regression
- Polynomial Linear Regression: `y = b0 + b1*x1 + b2*x1^(2) + ... +  bn*x1^(n)`
- Used for dataset with non-linear relation, but polynomial linear relation like salary scale.



[(Back to top)](#table-of-contents)
