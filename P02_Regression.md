# Regression
- [Table of contents](#table-of-contents)
- [Simple Linear Regression](#simple-linear-regression)
  - [Outline: Building a Model](#outline-building-a-model)
  - [Creating a Model](#creating-a-model)
  - [Predicting a Test Result](#predicting-a-test-result)
  - [Visualising the Test set results](#visualising-the-test-set-results)
  - [Getting Linear Regression Equation](#getting-linear-regression-equation)
  - [Evaluating the Algorithm](#evaluating-the-algorithm)
    - [R Square or Adjusted R Square](#r-square-or-adjusted-r-square)

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
<img src="https://user-images.githubusercontent.com/64508435/105186896-24174600-5b6d-11eb-9eb8-8c7f5e82d268.png" height="200" />


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
  1. R Square/Adjusted R Square
  2. Mean Square Error(MSE)/Root Mean Square Error(RMSE)
  3. Mean Absolute Error(MAE)

### R Square or Adjusted R Square
#### R Square: Coefficient of determination
- R Square measures how much of **variability** in predicted variable can be explained by the model.
- `Variance` is a measure in statistics defined as the average of the square of differences between individual point and the expected value.
- R Square value: between 0 to 1 and bigger value indicates a better fit between prediction and actual value.
- However, it does **not take into consideration of overfitting problem**. 
  - If your regression model has many independent variables, because the model is too complicated, it may fit very well to the training data 
  - but performs badly for testing data.
  - Solution: Adjusted R Square
<img src="https://user-images.githubusercontent.com/64508435/105422836-62048f00-5c7f-11eb-99f6-d94b3ebf1784.png" height="100" />
#### Adjusted R Square
- *Adjusted R Square* is introduced Since R-square can be increased by adding more number of variable and may lead to the **over-fitting** of the model
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

[(Back to top)](#table-of-contents)
