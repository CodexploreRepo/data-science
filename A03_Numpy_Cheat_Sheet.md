# Numpy Cheat Sheet
# Table of contents
- [Table of contents](#table-of-contents)
- [Introduction to Numpy](#introduction-to-numpy)

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
