# Numpy Cheat Sheet
# Table of contents
- [Table of contents](#table-of-contents)
- [Introduction to Numpy](#introduction-to-numpy)
- [Numpy Data Types and Attributes](numpy-data-types-and-attributes)

# Introduction to Numpy
### Why is Numpy important?
- How many decimal numbers we can store with `n bits` ? 
  - `n bits` is equal to 3 positions to store 0 & 1. 
  - Formula: 2^(n) = 8 decimal numbers
- Numpy allow you to specify more precisely number of memory you need for storing the data
```Python
#Python costs 28 bytes to store x = 5 since it is Integer Object
import sys
x = 5
sys.getsizeof(x) #return 28 - means variable x = 5 costs 28 bytes of memory

#Numpy : allow you to specify more precisely number of bits (memory) you need for storing the data
np.int8 #8-bit
```
- Numpy is **Array Processing**
  - Built-in DS in Python `List` NOT optimized for High-Level Processing as List in Python is Object and they will not store elements in separate position in Memory
  - In constrast, Numpy will store `Array Elements` in **Continuous Positions** in memory

### Numpy is  more efficient for storing and manipulating data

<img src="https://user-images.githubusercontent.com/64508435/108620727-1a814680-7469-11eb-8871-f8f2bd203a7d.png" height="250"/>

- `Numpy array` : essentially contains a single pointer to one contiguous block of data
- `Python list` : contains a pointer to a block of pointers, each of which in turn points to a full Python object

# Numpy Data Types and Attributes
- Main Numpy Data Type is `ndarray`
- Attributes: `shape, ndim, size, dtype`
