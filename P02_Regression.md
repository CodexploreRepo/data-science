# Regression
# Table of contents

- [Table of contents](#table-of-contents)
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

## Understand P value



[(Back to top)](#table-of-contents)
