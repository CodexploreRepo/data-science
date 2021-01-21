# Regression
- [Table of contents](#table-of-contents)
- [Simple Linear Regression](#simple-linear-regression)
  - [Creating a Model](#creating-a-model)

# Simple Linear Regression

```
y = bo + b1 * x1
```
- y: Dependent Variable (DV)
- x: InDependent Variable (IV)
- b: Coefficient

![Screenshot 2021-01-20 at 10 16 26 PM](https://user-images.githubusercontent.com/64508435/105186896-24174600-5b6d-11eb-9eb8-8c7f5e82d268.png)

## Creating a Model
- Using `sklearn.linear_model`,  `LinearRegression` model
```Python
from sklearn.linear_model import LinearRegression
#To Create Instance of Simple Linear Regression Model
regressor = LinearRegression()
#To fit the X_train and y_train
regressor.fit(X_train, y_train)
```



[(Back to top)](#table-of-contents)
