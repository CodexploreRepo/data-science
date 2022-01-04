# Kaggle 30 Days of Machine Learning
Resource: https://github.com/ageron/handson-ml

# Table of contents


  - [1.5. Cross-Validation](#15-cross-validation)
- [2. EDA](#2-eda)
  - [2.1. Graph](#21-graph) 
- [3. Feature Engineering](#3-feature-engineering)
- [4. Model Training](#4-model-training)
  - [4.1. Underfitting and Overfitting](#41-underfitting-and-overfitting)
  - [4.2. Evaluation Metrics](#42-evaluation-metrics)
    - [4.2.1. Metrics for Regression](#421-metrics-for-regression)
    - [4.2.2. Metrics for Classification](#422-metrics-for-classification)

- [6. Fine-Tune Model](#6-fine-tune-model)
  - [6.1. Grid Search](#61-grid-search)
  - [6.2. Randomized Search](#62-randomized-search)
  - [6.3. Analyze the Best Models and Their Errors](#63-analyze-the-best-models-and-their-errors)
  - [6.4. Evaluate Your System on the Test Set](#64-evaluate-your-system-on-the-test-set)
  - [6.5. Launch, Monitor, and Maintain Your System](#65-launch-monitor-and-maintain-your-system)
- [7. Save Model](#7-save-model)

## 1.4. Pipelines
## 1.5. Cross-Validation



# 4. Model Training
## 4.1. Underfitting and Overfitting


## 4.2. Evaluation Metrics
- Evaluation Metric used for Competition usually will be specified in Kaggle Competition > Evaluation 
<img width="951" alt="Screenshot 2021-08-15 at 16 25 45" src="https://user-images.githubusercontent.com/64508435/129472164-4101cc49-0320-4094-a9c4-8a5f697e30b6.png">




## 5.1 Ensemble methods
## 5.2. Stochastic Gradient Descent 
- `SGDClassifier` relies on randomness during training (hence the name “stochastic”)
- This classifier has the advantage of being capable of handling very large datasets efficiently. This is in part because SGD deals with training instances independently ((which also makes SGD well suited for online learning)

```Python
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
```

# Submission
```Python
predictions = model.predict(X_test)
output = pd.DataFrame({'id': test_data.id, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
```


[(Back to top)](#table-of-contents)


# 6. Fine-Tune Model

[(Back to top)](#table-of-contents)
