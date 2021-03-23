
# Matplotlib Cheat Sheet
# Table of contents
- [Table of contents](#table-of-contents)
- [Introduction to Matplotlib](#introduction-to-matplotlib)


# Introduction to Matplotlib
<img width="998" alt="matplotlib-anatomy-of-a-plot" src="https://user-images.githubusercontent.com/64508435/112073781-b5c42380-8baf-11eb-87db-f4241ea7232a.png">

- `Figure` can contains multiple Subplot
- `Axes 0` and `Axes 1` stacked together
## Pyplot API vs Object-oriented API
- In general, try to use the `object-oriented interface` (more flexible) over the `pyplot` interface (i.e: `plt.plot()`

```Python

x = [1,2,3,4]
y = [11,22,33,44]

# Pyplot API
plt.plot(x,y)

# [Recommended] Object-oriented interface 
fig, ax = plt.subplots() #create figure + set of subplots, by default, nrow =1, ncol=1
ax.plot(x,y) #add some data
plt.show()
```
