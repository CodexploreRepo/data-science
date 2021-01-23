
# Pandas Cheat Sheet
# Table of contents
- [Table of contents](#table-of-contents)
- [Getting and knowing](#getting-and-knowing)
  - [Access rows or Columns of Data Frame](#access-rows-or-columns-of-data-frame)
- [Grouping](#grouping)
  - [Basic Grouping](#basic-grouping)

# Getting and knowing
### shape : Return (Row, Column)
```Python
df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4],
                   'col3': [5, 6]})
df.shape
(2, 3) # df.shape[0] = 2 row, df.shape[1] = 3 col
```
### info() : Return index dtype, columns, non-null values & memory usage.
```Python
df.info()
```
- We will understand dtype of cols, how many non-null value of DF
```Python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4622 entries, 0 to 4621
Data columns (total 5 columns):
 #   Column              Non-Null Count  Dtype 
---  ------              --------------  ----- 
 0   order_id            4622 non-null   int64 
 1   quantity            4622 non-null   int64 
 2   item_name           4622 non-null   object
 3   choice_description  3376 non-null   object
 4   item_price          4622 non-null   object
dtypes: int64(2), object(3)
memory usage: 180.7+ KB
````
### dtype : Return data type of specific column
- `df.col_name.dtype` return the data type of that column
```Python
df.item_price.dtype
#'O'     (Python) objects
```

- Please note: dtype will return below special character
```
'b'       boolean
'i'       (signed) integer
'u'       unsigned integer
'f'       floating-point
'c'       complex-floating point
'O'       (Python) objects
'S', 'a'  (byte-)string
'U'       Unicode
'V'       raw data (void
```

## Access Rows or Columns of Data Frame
### Columns
#### Print the name of all the columns
```Python
list(df.columns)
#['order_id', 'quantity', 'item_name', 'choice_description','item_price', 'revenue']
```
#### Access column
```Python
# Counting how many values in the column
df.col_name.count()
# Take the mean of values in the column
df["col_name"].mean()
```
##### value_counts() : Return a Series containing counts of unique values
Ex 1: Calculate unique values in a columns
```
index = pd.Index([3, 1, 2, 3, 4, np.nan])
index.value_counts()
#Return:
3.0    2
2.0    1
4.0    1
1.0    1
dtype: int64
#How many unique values 
index.value_counts().count()
#4

```

#### Check index of DF
```Python
df.index
#RangeIndex(start=0, stop=4622, step=1)
```
[(Back to top)](#table-of-contents)

# Grouping
<img width="707" alt="Screenshot 2021-01-23 at 10 47 21 PM" src="https://user-images.githubusercontent.com/64508435/105590696-195aec00-5dcd-11eb-894d-3953d6ea8180.png">

## Basic Grouping
- Grouping by "item_name" column & take the sum of "quantity" column
- Method #1 : `df.groupby("item_name")`

```Python
df.groupby("item_name")["quantity"].sum()
```

```Python
item_name
Chicken Bowl       761
Chicken Burrito    591
Name: quantity, dtype: int64
```

- Method #2: df.groupby(by=['order_id'])

```Python
order_revenue = df.groupby(by=["order_id"])["revenue"].sum()
```
[(Back to top)](#table-of-contents)




