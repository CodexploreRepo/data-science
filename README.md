# Data Science Handbook
Data Science Handbook

# Data Science Handbook
# Table of contents

- [Table of contents](#table-of-contents)
- [Introduction](#introduction)
- [Data Preprocessing](#Data-Preprocessing)
  - [Import Dataset](#import-dataset)
  - [Select Data](#select-data)
    - [Using Index: iloc](#using-index-iloc)
  - [Numpy representation of DF](#Numpy-representation-of-DF)
- [Resources](#resources)


# Introduction 


[(Back to top)](#table-of-contents)

# Data Preprocessing 
## Import Dataset
```python
dataset = pd.read_csv("data.csv")

   Country   Age   Salary Purchased
0   France  44.0  72000.0        No
1    Spain  27.0  48000.0       Yes
2  Germany  30.0  54000.0        No
3    Spain  38.0  61000.0        No
4  Germany  40.0      NaN       Yes
5   France  35.0  58000.0       Yes
6    Spain   NaN  52000.0        No
7   France  48.0  79000.0       Yes
8  Germany  50.0  83000.0        No
9   France  37.0  67000.0       Yes
```
## Select Data
### Using Index iloc
- `.iloc[]` allowed inputs are:
  #### Selecting Rows
  - An integer, e.g. `dataset.iloc[0]` > return row 0 in Series
  ```
  Country      France
  Age              44
  Salary        72000
  Purchased        No
  ```
  - A list or array of integers, e.g.`dataset.iloc[[0]]` > return row 0 in DataFrame format
  ```
     Country   Age   Salary  Purchased
  0  France    44.0  72000.0        No
  ```
  - A slice object with ints, e.g. `dataset.iloc[:3]` > return row 0 up to row 3 in DataFrame format
  ```
       Country   Age   Salary Purchased
  0    France   44.0  72000.0        No
  1    Spain    27.0  48000.0       Yes
  2    Germany  30.0  54000.0        No
  ```
  #### Selecting Rows & Columns
  - Select First 3 Rows & up to Last Columns (not included) `X = dataset.iloc[:3, :-1]`
  ```
       Country   Age   Salary
  0   France  44.0  72000.0
  1    Spain  27.0  48000.0
  2  Germany  30.0  54000.0
  ```
### Numpy representation of DF
- `DataFrame.values`: Return a Numpy representation of the DataFrame (i.e: Only the values in the DataFrame will be returned, the axes labels will be removed)
- For ex: `X = dataset.iloc[:3, :-1].values`
```
[['France' 44.0 72000.0]
 ['Spain' 27.0 48000.0]
 ['Germany' 30.0 54000.0]]
```
# Resources:
### Podcast:
https://www.superdatascience.com/podcast/sds-041-inspiring-journey-totally-different-background-data-science




