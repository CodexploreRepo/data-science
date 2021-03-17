
# Pandas Cheat Sheet
# Table of contents
- [Table of contents](#table-of-contents)
- [Import & Export Data](#import-export-data)
- [Getting and knowing](#getting-and-knowing)
  - [loc vs iloc](#loc-vs-iloc)
  - [Access Rows of Data Frame](#access-columns-of-data-frame)
  - [Access Columns of Data Frame](#access-columns-of-data-frame)
- [Manipulating Data](#manipulating-data)
- [Grouping](#grouping)
  - [Basic Grouping](#basic-grouping)


# Import Export Data
### Import with Different Separator
```Python
users = pd.read_csv('user.csv', sep='|')
chipo = pd.read_csv(url, sep = "\t")
```
<img height="500" alt="pandas-anatomy-of-a-dataframe" src="https://user-images.githubusercontent.com/64508435/111490410-f833cd80-8775-11eb-8527-daf08dc8e91a.png">

#### Renaming Index
```Python
users = pd.read_csv('u.user', sep='|', index_col='user_id')
```
### Export 
```Python
users.to_csv("exported-users.csv")
```

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

### describe() : Generate descriptive statistics.
```Python
chipo.describe() #Notice: by default, only the numeric columns are returned. 
chipo.describe(include = "all") #Notice: By default, only the numeric columns are returned.
```


### dtype : Return data type of specific column
- `df.col_name.dtype` return the data type of that column
```Python
df.item_price.dtype
#'O'     (Python) objects
```

- Please note: dtype will return below special character
```Python
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

## loc vs iloc
### loc
- `loc`: is **label-based**, which means that we have to specify the "name of the rows and columns" that we need to filter out.
#### Find all the rows based on 1 or more conditions in a column
```Python
# select all rows with a condition
data.loc[data.age >= 15]
# select all rows with multiple conditions
data.loc[(data.age >= 12) & (data.gender == 'M')]
```
![image](https://user-images.githubusercontent.com/64508435/106067849-7abaec00-613a-11eb-8cbe-f9aa5e2c6202.png)

#### Select only required columns with conditions
```Python
# Update the values of multiple columns on selected rows
chipo.loc[(chipo.quantity == 7) & (chipo.item_name == 'Bottled Water'), ['item_name', 'item_price']] = ['Tra Xanh', 0]
# Select only required columns with a condition
chipo.loc[(chipo.quantity > 5), ['item_name', 'quantity', 'item_price']]
```
<img width="381" alt="Screenshot 2021-01-28 at 7 26 04 AM" src="https://user-images.githubusercontent.com/64508435/106067706-32033300-613a-11eb-98ce-114c4c0f9dd6.png">

### iloc
- `iloc`: is **index-based**, which means that we have to specify the "integer index-based" that we need to filter out.
- `.iloc[]` allowed inputs are:
  #### Selecting Rows
  - An integer, e.g. `dataset.iloc[0]` > return row 0 in `<class 'pandas.core.series.Series'>`
  ```Python
  Country      France
  Age              44
  Salary        72000
  Purchased        No
  ```
  - A list or array of integers, e.g.`dataset.iloc[[0]]` > return row 0 in DataFrame format
  ```Python
     Country   Age   Salary  Purchased
  0  France    44.0  72000.0        No
  ```
  - A slice object with ints, e.g. `dataset.iloc[:3]` > return row 0 up to row 3 in DataFrame format
  ```Python
       Country   Age   Salary Purchased
  0    France   44.0  72000.0        No
  1    Spain    27.0  48000.0       Yes
  2    Germany  30.0  54000.0        No
  ```
  #### Selecting Rows & Columns
  - Select First 3 Rows & up to Last Columns (not included) `X = dataset.iloc[:3, :-1]`
  ```Python
       Country   Age   Salary
  0   France  44.0  72000.0
  1    Spain  27.0  48000.0
  2  Germany  30.0  54000.0
  ```
### Numpy representation of DF
- `DataFrame.values`: Return a Numpy representation of the DataFrame (i.e: Only the values in the DataFrame will be returned, the axes labels will be removed)
- For ex: `X = dataset.iloc[:3, :-1].values`
```Python
[['France' 44.0 72000.0]
 ['Spain' 27.0 48000.0]
 ['Germany' 30.0 54000.0]]
```

## Access Rows of Data Frame
### Check index of DF
```Python
df.index
#RangeIndex(start=0, stop=4622, step=1)
```

[(Back to top)](#table-of-contents)

## Access Columns of Data Frame
### Print the name of all the columns
```Python
list(df.columns)
#['order_id', 'quantity', 'item_name', 'choice_description','item_price', 'revenue']
```
### Access column
```Python
# Counting how many values in the column
df.col_name.count()
# Take the mean of values in the column
df["col_name"].mean()
```
### value_counts() : Return a Series containing counts of unique values
```Python
index = pd.Index([3, 1, 2, 3, 4, np.nan])
#dropna=False will also consider NaN as a unique value 
index.value_counts(dropna=False)
#Return: 
3.0    2
2.0    1
NaN    1
4.0    1
1.0    1
dtype: int64
```
### Calculate total unique values in a columns
```Python
#How many unique values 
index.value_counts().count()

index.nunique()
#5
```

[(Back to top)](#table-of-contents)
# Manipulating Data
## Missing Values
### Filling Missing Values with fillna()
- To fill `nan` value with a v
```Python
car_sales_missing["Odometer"].fillna(car_sales_missing["Odometer"].mean(), inplace = True)
```
### Dropping Missing Values with dropna()
- To drop columns containing Missing Values
```Python
car_sales_missing.dropna(inplace=True)
```
## Drop a column

```Python
car_sales.drop("Passed road safety", axis = 1) # axis = 1 if you want to drop a column
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

- Method #2: `df.groupby(by=['order_id'])`

```Python
order_revenue = df.groupby(by=["order_id"])["revenue"].sum()
```
[(Back to top)](#table-of-contents)




