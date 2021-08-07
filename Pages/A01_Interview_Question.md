# Data Scientist - Interview Questions

## Table of contents
- [1. Interview Questions](#1-interview-questions)
- [2. SQL Questions](#2-sql-questions)


## 1. Interview Questions
### 1.1. Data Scientist
- **[Facebook] Product Generalist** (i.e. solving a business case study)
  - How to design the friends you may know feature -- how to recommend friends; 
  - How can we tell if two users on Instagram are best friends
  - How would you build a model to decide what to show a user and how would you evaluate its success
  - How would you create a model to find bad sellers on marketplace?
- **[Facebook] Coding Exercise (in SQL)**: joins (LEFT, RIGHT, UNION), group by, date manipulation
- **[Facebook] Quantitative Analysis**
  - How to test out the assumptions; how to decide next steps if the metrics shows only positive signals in certain features  
  - How can you tell if your model is working?
- **[Facebook] Applied Data (stats questions)**: AB Testing

### 1.2. Machine Learning Engineer
- **Coding Interview**: 
  - [Data Structure] Difference Stack vs Queue, Dequeue Implementation, Linked List Reversal
  - [Easy] Reverse a linked list, Convert decimal to hexadecimal without using built-in methods (str, int etc.), pairs of number that sum up to K
  - [Medium] Verify binary search tree
  - [Hard] Min edit distance
- **Techinical Interview**: 
  - Fundamental ML questions: 
    - Non-deep and deep methods
    - Basic ML models or algorithms: Formula for gradient descent, Linear an Non-Linear classifiers, K-means, Random forest, Clustering Nearest neighbors. Decision Tree
    - Basic DL: Explain how CNN works, Recurrent neural network
    - Metric Understanding: ROC
    - What is overfitting?  
    - Difference between Bagging and Boosting 
    - Regularization: Diff of L1 and L2 regularization
  - System Design: 
    - How to search efficiently
    - Given salaries of people from ten professions and salary of a new people. Design an algorithm to predict the profession of this new people.
  - Case Study:
    - How would you apply A/B testing on food odering service
    - How surge pricing works for both customers and drivers
    - Implement Huffman code for a given English sentence  
- **Interview with Hiring Manager**: explain your Machine learning projects

### 1.3. Grab: Machine Learning Engineer

## 2. SQL Questions
#### SQL#1: Facebook
```SQL
Given the following data:

Table:
searches
Columns:
date STRING date of the search,
search_id INT the unique identifier of each search,
user_id INT the unique identifier of the searcher,
age_group STRING ('<30', '30-50', '50+'),
search_query STRING the text of the search query

Sample Rows:
date | search_id | user_id | age_group | search_query
--------------------------------------------------------------------
'2020-01-01' | 101 | 9991 | '<30' | 'justin bieber'
'2020-01-01' | 102 | 9991 | '<30' | 'menlo park'
'2020-01-01' | 103 | 5555 | '30-50' | 'john'
'2020-01-01' | 104 | 1234 | '50+' | 'funny cats'


Table:
search_results
Columns:
date STRING date of the search action,
search_id INT the unique identifier of each search,
result_id INT the unique identifier of the result,
result_type STRING (page, event, group, person, post, etc.),
clicked BOOLEAN did the user click on the result?

Sample Rows:
date | search_id | result_id | result_type | clicked
--------------------------------------------------------------------
'2020-01-01' | 101 | 1001 | 'page' | TRUE
'2020-01-01' | 101 | 1002 | 'event' | FALSE
'2020-01-01' | 101 | 1003 | 'event' | FALSE
'2020-01-01' | 101 | 1004 | 'group' | FALSE


Over the last 7 days, how many users made more than 10 searches?

You notice that the number of users that clicked on a search result
about a Facebook Event increased 10% week-over-week. How would you
investigate? How do you decide if this is a good thing or a bad thing?

The Events team wants to up-rank Events such that they show up higher
in Search. How would you determine if this is a good idea or not?
```
[(Back to top)](#table-of-contents)
