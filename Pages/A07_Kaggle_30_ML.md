# Kaggle 30 Days of Machine Learning

# Table of contents
- [1. Data Pre-Processing](#1-data-pre-processing)
- [2. EDA](#2-eda)
  - [2.1. Graph](#21-graph) 
- [3. Feature Engineering](#3-feature-engineering)
- [4. Model Training](#4-model-training)
  - [4.1. Underfitting and Overfitting](#41-underfitting-and-overfitting)

# 1. Data Pre-Processing

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

# Submission
```Python
predictions = model.predict(X_test)
output = pd.DataFrame({'id': test_data.id, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
```


[(Back to top)](#table-of-contents)
