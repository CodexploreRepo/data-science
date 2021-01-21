# Regression
- [Table of contents](#table-of-contents)
- [Simple Linear Regression](#simple-linear-regression)
  - [Outline: Building a Model](#outline-building-a-model)
  - [Creating a Model](#creating-a-model)
  - [Predicting a Test Result](#predicting-a-test-result)
  - [Visualising the Test set results](#visualising-the-test-set-results)
  - [Getting Linear Regression Equation](#getting-linear-regression-equation)

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
- b: Coefficient
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
plt.scatter(X_test, y_test, color = 'red')
#Plot the regression line
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
#Label the Plot
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
#Show the plot
plt.show()
```
![download](https://user-images.githubusercontent.com/64508435/105365689-7c1b7e80-5c39-11eb-8e44-12866fb7eb3d.png)

## Getting Linear Regression Equation
```Python
print(f"b0 : {regressor.intercept_}")
print(f"b1 : {regressor.coef_}")

b0 : 25609.89799835482
b1 : [9332.94473799]
```

Linear Regression Equation: `Salary = 9332.94×YearsExperience + 25609`

[(Back to top)](#table-of-contents)
