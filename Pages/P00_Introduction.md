# Introduction
# Table of contents
- [Table of contents](#table-of-contents)
- [Why need to learn Machine Learning ?](#why-need-to-learn-machine-learning)
- [Terms](#terms)
  - [AI](#ai)
  - [Machine Learning](#machine-learning)
  - [Deep Learning](#deep-learning)
  - [Data Science](#data-science)
- [Machine Learning Framework](#machine-learning-framework)
  - [Main Types of ML Problems](#main-types-of-ml-problems) 
  - [Evaluation](#evaluation)
  - [Features](#features) 
  - [Modelling](#modelling)
    - [Splitting Data](#splitting-data)
    - [Modelling](#modelling)
    - [Tuning](#tuning)
    - [Comparison](#comparison)

# Why need to learn Machine Learning ?
<img src="https://user-images.githubusercontent.com/64508435/109885685-e8b67e00-7cb9-11eb-9d62-40fe7f1ff66a.png" height="350px"/>

- **Spread Sheets (Excel, CSV)**: store data that business needs → Human can analyse data to make business decision
- **Relational DB (MySQL)**: a better way to organize things → Human can analyse data to make business decision
- **Big Data (NoSQL)**: FB, Amazon, Twitter accumulating more and more data like "User actions, user purchasing history", where you can store un-structure data → need Machine Learning instead of Human to make business decision

[(Back to top)](#table-of-contents)

# Terms
## AI
## Machine Learning

<img src="https://user-images.githubusercontent.com/64508435/110039927-f254ea00-7d7c-11eb-9ff4-52b498925232.png" height="150px" />

- [A subset of AI](https://teachablemachine.withgoogle.com/): ML uses Algorithms or Computer Programs to learn different patterns of data & then take those algorithms & what it learned to make prediction or classification on similar data.
- The things hard to describe for computers to perform like 
  - How to ask Computers to classify Cat/Dog images, or Product Reviews

### Difference between ML and Normal Algorithms
- Normal Algorithm: a set of instructions on how to accomplish a task: start with `given input + set of instructions` → output
- ML Algorithm    : start with `given input + given output` → set of instructions between I/P and O/P
<img src="https://user-images.githubusercontent.com/64508435/110040442-cc7c1500-7d7d-11eb-87e6-cc4583b7aec4.png" height="400px" />

### Types of ML Problems

<img src="https://user-images.githubusercontent.com/64508435/109983163-8a32e380-7d3d-11eb-85fd-a2635e14826c.png" height="400px"/>

- **Supervised**: Data with Label
- **Unsupervised**: Data without Label like CSV without Column Names
  - *Clustering*: Machine decicdes clusters/groups
  - *Association Rule Learning*: Associate different things to predict what customers might buy in the future
- **Reinforcement**: teach Machine to try and error (with reward and penalty)

## Deep Learning
## Data Science
- `Data Analysis`: analyse data to gain understanding of your data
- `Data Science` : running experiments on set of data to figure actionable insights within it
  - Example: to build ML Models

[(Back to top)](#table-of-contents)

# Machine Learning Framework
![Screenshot 2021-03-05 at 7 00 17 AM](https://user-images.githubusercontent.com/64508435/110042238-7361b080-7d80-11eb-825d-f8fc4d4c2cf2.png)

- Readings: [ (1) ](https://www.mrdbourke.com/a-6-step-field-guide-for-building-machine-learning-projects/), [ (2) ](https://whimsical.com/6-step-field-guide-to-machine-learning-projects-flowcharts-9g65jgoRYTxMXxDosndYTB)
### Step 1: Problem Definition - Rephrase business problem as a machine learning problem
- What problem are we trying to solve ?
  - Supervised
  - Un-supervised
  - Classification
  - Regression
### Step 2: Data
- What kind of Data we have ? 
### Step 3: Evaluation
- What defines success for us ? knowing what metrics you should be paying attention to gives you an idea of how to evaluate your machine learning project.
### Step 4: Features
- What features does your data have and which can you use to build your model ? turning features → patterns
- **Three main types of features**: 
  - `Categorical` features — One or the other(s) 
    - For example, in our heart disease problem, the sex of the patient. Or for an online store, whether or not someone has made a purchase or not.
  - `Continuous (or numerical)` features: A numerical value such as average heart rate or the number of times logged in.
  - `Derived` features — Features you create from the data. Often referred to as feature engineering. 
    - `Feature engineering` is how a subject matter expert takes their knowledge and encodes it into the data. You might combine the number of times logged in with timestamps to make a feature called time since last login. Or turn dates from numbers into “is a weekday (yes)” and “is a weekday (no)”.
### Step 5: Models
- Figure out right models for your problems
### Step 6: Experimentation
- How to improve or what can do better ?

## Main Types of ML Problems
![Screenshot 2021-03-09 at 8 23 37 AM](https://user-images.githubusercontent.com/64508435/110399393-c1dcbb00-80b0-11eb-8c0d-4b21f02fc3e4.png)
### Supervised Learning:
- (Input & Output) Data + Label → Classifications, Regressions
### Un-Supervised Learning:
- (Only Input) Data → Clustering
### Transfer Learning:
- (My problem similar to others) Leverage from Other ML Models
### Reinforcement Learning:
- Purnishing & Rewarding the ML Learning model by updating the scores of ML 

## Evaluation

| Classification     |  Regression                        | Recommendation  |
| -------------------| ---------------------------------- | ----------------|
| Accuracy           | Mean Absolute Error (MAE)          |  Precision at K |
| Precision          | Mean Squared Error (MSE)           |    |
| Recall             | Root Mean Squared Error (RMSE)     |    |

[(Back to top)](#table-of-contents)

## Features 
- Numerical Features
- Categorical Features

<img src="https://user-images.githubusercontent.com/64508435/110710402-c03dff00-8238-11eb-96c6-592fd9be4076.png" height="350px" />

[(Back to top)](#table-of-contents)

## Modelling
### Splitting Data

<img src="https://user-images.githubusercontent.com/64508435/110711094-eadc8780-8239-11eb-91e4-04657d7ed079.png" height="200px" />

- 3 sets: Trainning, Validation (model hyperparameter tuning and experimentation evaluation) & Test Sets (model testing and comparison)

### Modelling
- Chosen models work for your problem  → train the model 
- Goal: Minimise time between experiments 
  - Start small  and add up complexity (use small parts of your training sets to start with)
  - Choosing the less complicated models to start first
<img src="https://user-images.githubusercontent.com/64508435/110711505-a30a3000-823a-11eb-9dfd-eb5283da7720.png" height="200px" />

### Tuning 
- Happens on Validation or Training Sets

### Comparison
- Measure Model Performance via Test Set
- Advoid `Overfitting` & `Underfitting`
#### Overfitting
- Great performance on the training data but poor performance on test data means your model doesn’t generalize well
- Solution: Try simpler model or making sure your the test data is of the same style your model is training on
### Underfitting
- Poor performance on training data means the model hasn’t learned properly and is underfitting
- Solution: Try a different model, improve the existing one through hyperparameter or collect more data.

<img src="https://user-images.githubusercontent.com/64508435/110889708-a4feec80-8329-11eb-8399-a02ae4274002.png" height="200px" />

