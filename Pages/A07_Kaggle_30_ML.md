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
- [5. Machine Learning Model](#5-machine-learning-model)
  - [5.1 Ensemble methods](#51-ensemble-methods)
    - [5.1.1 Random Forests](#511-random-forests)
    - [5.2.2 Gradient Boosting](#512-gradient-boosting)
  - [5.2. Stochastic Gradient Descent](#52-stochastic-gradient-descent) 
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
- The group (or “ensemble”) will often perform better than the best individual model (just like Random Forests perform better than the individual Decision Trees they rely on), especially if the individual models make very different types of errors. (more on [Chapter 7](https://www.oreilly.com/library/view/hands-on-machine-learning/9781491962282/ch07.html#ensembles_chapter))
- The goal of `ensemble methods` is to combine the predictions of several base estimators built with a given learning algorithm in order to improve generalizability / robustness over a single estimator (**for classification, regression and anomaly detection**)
- Two families of ensemble methods:
  - In **averaging methods**, the driving principle is to build several estimators independently and then to average their predictions. On average, the combined estimator is usually better than any of the single base estimator because its variance is reduced.
    - *Examples*: Bagging methods, Forests of randomized trees, etc.
  - In **boosting methods**, base estimators are built sequentially and one tries to reduce the bias of the combined estimator. The motivation is to combine several weak models to produce a powerful ensemble.
    - *Examples*: AdaBoost, Gradient Tree Boosting, etc.

### 5.1.1 Random Forests
- Random forest method, which achieves better performance than a single decision tree simply by averaging the predictions of many decision trees. We refer to the random forest method as an "ensemble method". By definition, ensemble methods combine the predictions of several models (e.g., several trees, in the case of random forests).
- Decision trees leave you with a difficult decision. 
  - A deep tree with lots of leaves will overfit because each prediction is coming from historical data from only the few data at its leaf. 
  - But a shallow tree with few leaves will perform poorly because it fails to capture as many distinctions in the raw data.
- The random forest uses many trees, and it makes a prediction by averaging the predictions of each component tree. 
- It generally has much better predictive accuracy than a single decision tree and it works well with default parameters.

### 5.1.2. Gradient Boosting
- **Gradient boosting** is the method dominates many Kaggle competitions and achieves state-of-the-art results on a variety of datasets.
- Gradient boosting is a method that goes through cycles to iteratively add models into an ensemble.
- It begins by initializing the ensemble with a single model, whose predictions can be pretty naive. (Even if its predictions are wildly inaccurate, subsequent additions to the ensemble will address those errors.)
- *"Gradient"* in "gradient boosting" refers to the fact that we'll use `gradient descent` on the loss function to determine the parameters in this new model.)
![image](https://user-images.githubusercontent.com/64508435/144753278-7e6573ec-4a2e-45e9-aa2b-b713529366d0.png)


```Python
#Example of Gradient Boosting - Regressor
from sklearn.ensemble import GradientBoostingRegressor

gbm_model = GradientBoostingRegressor(random_state=1, n_estimators=500)
gbm_model.fit(train_X, train_y)
gbm_val_predictions = gbm_model.predict(val_X)
gbm_val_rmse = np.sqrt(mean_squared_error(gbm_val_predictions, val_y))
```

### 5.1.2.1. XGBoost 
- `XGBoost` stands for **extreme gradient boosting**, which is an implementation of gradient boosting with several additional features focused on performance and speed.
- XGBoost has a few parameters that can dramatically affect accuracy and training speed:
  - `n_estimators`: (typically range from **100-1000**, though this depends a lot on the learning_rate parameter) specifies how many times to go through the modeling cycle described above. It is equal to the number of models that we include in the ensemble.
    - Too low a value causes *underfitting*, which leads to inaccurate predictions on both training data and test data.
    - Too high a value causes *overfitting*, which causes accurate predictions on training data, but inaccurate predictions on test data (which is what we care about).
  - `learning_rate`: (default: learning_rate=0.1) Instead of getting predictions by simply adding up the predictions from each component model, we can multiply the predictions from each model by a small number (known as the **learning rate**) before adding them in.
  - `n_jobs`: equal to the number of cores on your machine
    - On smaller datasets, the resulting model won't be any better, so micro-optimizing for fitting time is typically nothing but a distraction.
    - On larger datasets where runtime is a consideration, you can use parallelism to build your models faster; otherwise spend a long time waiting during the `fit` command.
```Python
from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4, random_state = 42)

my_model.fit(X_train, y_train, 
             #Setting early_stopping_rounds=5 is a reasonable choice. In this case, we stop after 5 straight rounds of deteriorating validation scores
             early_stopping_rounds=5, 
             #When using early_stopping_rounds, you also need to set aside some data for calculating the validation scores - this is done by setting the eval_set parameter.
             eval_set=[(X_valid, y_valid)], 
             verbose=False)
```
  - `early_stopping_rounds`: offers a way to automatically find the ideal value for n_estimators. Early stopping causes the model to stop iterating when the validation score stops improving, even if we aren't at the hard stop for n_estimators. It's smart to set a high value for n_estimators and then use early_stopping_rounds to find the optimal time to stop iterating.
    - Setting **early_stopping_rounds=5** is a reasonable choice. In this case, we stop after 5 straight rounds of deteriorating validation scores.


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

Let’s assume that you now have a shortlist of promising models. You now need to fine-tune them.

## 6.1. Grid Search
- One way to do that would be to fiddle with the hyperparameters manually, until you find a great combination of hyperparameter
- Scikit-Learn’s `GridSearchCV` to search which hyperparameters you want it to experiment with, and what values to try out 
- It will evaluate all the possible combinations of hyperparameter values, using cross-validation

```Python
from sklearn.model_selection import GridSearchCV

param_grid = [
    #first evaluate all 3 × 4 = 12 combinations of n_estimators and max_features hyperparameter values specified in the first dict
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    #all 2 × 3 = 6 combinations of hyperparameter values in the second dict
    #this time with the bootstrap hyperparameter set to False instead of True (which is the default)
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor()

#All in all, the grid search will explore 12 + 6 = 18 combinations of RandomForestRegressor hyperparameter values
#Train each model five times (since we are using five-fold cross validation)
#All in all, there will be 18 × 5 = 90 rounds of training
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

grid_search.fit(housing_prepared, housing_labels)
```
-  Tip: Since 8 and 30 are the maximum values that were evaluated, you should probably try searching again with higher values, since the score may continue to improve.
```Python
#get the best combination of parameters
grid_search.best_params_
```
- If `GridSearchCV` is initialized with `refit=True` (which is the default), then once it finds the best estimator using cross-validation, it retrains it on the whole training set. This is usually a good idea since feeding it more data will likely improve its performance.

```Python
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

#50036.32733962357 {'max_features': 8, 'n_estimators': 30}
#61747.39782442657 {'bootstrap': False, 'max_features': 2, 'n_estimators': 3}
```

## 6.2. Randomized Search
- When the hyperparameter search space is large, it is often preferable to use `RandomizedSearchCV` instead. 
- Instead of trying out all possible combinations like `GridSearchCV`, it evaluates a given number of random combinations by selecting a random value for each hyperparameter at every iteration. 
- This approach has two main benefits:
  - If you let the randomized search run for, say, 1,000 iterations, this approach will explore 1,000 different values for each hyperparameter (instead of just a few values per hyperparameter with the grid search approach).
  - You have more control over the computing budget you want to allocate to hyperparameter search, simply by setting the number of iterations.

```Python
from sklearn.model_selection import RandomizedSearchCV

# Setup random seed
np.random.seed(42)

param_grid = [
    #first evaluate all 3 × 4 = 12 combinations of n_estimators and max_features hyperparameter values specified in the first dict
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    #all 2 × 3 = 6 combinations of hyperparameter values in the second dict
    #this time with the bootstrap hyperparameter set to False instead of True (which is the default)
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

# Setup random hyperparameter search for RandomForestClassifier
randomized_search = RandomizedSearchCV(forest_reg, 
                           param_distributions=param_grid,
                           cv=5,
                           n_iter=10,
                           scoring='neg_mean_squared_error',
                           return_train_score=True,
                                      verbose=True)

# Fit random hyperparameter search model for RandomForestClassifier()
randomized_search.fit(housing_prepared, housing_labels)
```
- To get the best combination of parameters
```Python
#get the best combination of parameters
randomized_search.best_params_
#{'n_estimators': 30, 'max_features': 8}

#get the score for each combination
cvres = randomized_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

#65029.23239964716 {'n_estimators': 3, 'max_features': 2}
#55289.066755389285 {'n_estimators': 10, 'max_features': 2}
```

## 6.3. Analyze the Best Models and Their Errors
- You will gain good insights on the problem by inspecting the best models. 
    - For example, the `RandomForestRegressor` can indicate the relative importance of each attribute for making accurate predictions
```Python
#To get the feature importances of the best estimator
feature_importances = randomized_search.best_estimator_.feature_importances_

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])

attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)

"""
[(0.34494467079972574, 'median_income'),
 (0.17270875334179428, 'INLAND'),
 (0.11156438666412408, 'pop_per_hhold'),
 (0.0699146551283051, 'bedrooms_per_room'),
 (0.0680788476897446, 'longitude'),
 (0.06437338799728325, 'latitude'),
 (0.05229664262496094, 'rooms_per_hhold'),
 (0.04300665520810458, 'housing_median_age'),
 (0.01624609185547273, 'total_rooms'),
 (0.015168543028308725, 'population'),
 (0.014458910047585575, 'total_bedrooms'),
 (0.014129424198288248, 'households'),
 (0.007866351833047272, '<1H OCEAN'),
 (0.0030872145999579397, 'NEAR OCEAN'),
 (0.0020850563416243396, 'NEAR BAY'),
 (7.0408641672674e-05, 'ISLAND')]
"""
```
- With this information, you may want to try dropping some of the less useful features (e.g., apparently only one `ocean_proximity ( 'INLAND')` category is really useful, so you could try dropping the others `('<1H OCEAN','ISLAND','NEAR BAY', 'NEAR OCEAN')`).

## 6.4. Evaluate Your System on the Test Set
- After tweaking your models for a while, you eventually have a system that performs sufficiently well. 
- Now is the time to evaluate the final model on the test set.
- get the predictors and the labels from your test set, run your full_pipeline to transform the data (call transform(), not fit_transform(), you do not want to fit the test set!), and evaluate the final model on the test set:

```Python
final_model = randomized_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)   # => evaluates to 48006
final_rmse
```

## 6.5. Launch, Monitor, and Maintain Your System
- Now, You need to get your solution ready for production, in particular by plugging the production input data sources into your system and writing tests.
- You also need to write monitoring code to check your system’s live performance at regular intervals and trigger alerts when it drops.
  - This is important to catch not only sudden breakage, but also performance degradation. 
- You should also make sure you evaluate the system’s input data quality.
  - Sometimes performance will degrade slightly because of a poor quality signal (e.g., a malfunctioning sensor sending random values, or another team’s output becoming stale)
- You will generally want to train your models on a regular basis using fresh data. You should automate this process as much as possible. 
  - If your system is an online learning system, you should make sure you save snapshots of its state at regular intervals so you can easily roll back to a previously working state.

[(Back to top)](#table-of-contents)

# 7. Save Model
- You can easily save Scikit-Learn models by using Python’s `pickle` module, or using `sklearn.externals.joblib`, which is more efficient at serializing large NumPy arrays:

```Python
from sklearn.externals import joblib

joblib.dump(my_model, "my_model.pkl")

# and later...
my_model_loaded = joblib.load("my_model.pkl")
```
[(Back to top)](#table-of-contents)
