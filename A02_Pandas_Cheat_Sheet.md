
# Pandas Cheat Sheet
# Table of contents
- [Table of contents](#table-of-contents)
- [Getting and knowing](#getting-and-knowing)
  - [Access rows or Columns of Data Frame](#access-rows-or-columns-of-data-frame)

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
## Access Rows or Columns of Data Frame
#### Print the name of all the columns
```Python
list(df.columns)
#['order_id', 'quantity', 'item_name', 'choice_description','item_price', 'revenue']
```
#### Check index of DF
```Python
df.index
#RangeIndex(start=0, stop=4622, step=1)
```
[(Back to top)](#table-of-contents)




