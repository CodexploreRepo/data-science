# Kaggle 30 Days of Machine Learning

# Table of contents
- [1. Data Pre-Processing](#1-data-pre-processing)
- [2. EDA](#2-eda)
  - [2.1. Graph](#21-graph) 



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

# Submission
```Python
predictions = model.predict(X_test)

output = pd.DataFrame({'id': test_data.id, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
```


[(Back to top)](#table-of-contents)
