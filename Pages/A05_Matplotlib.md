
# Matplotlib Cheat Sheet
# Table of contents
- [Table of contents](#table-of-contents)
- [Introduction to Matplotlib](#introduction-to-matplotlib)
  - [Matplotlib Workflow](#matplotlib-workflow)  
- [Subplots](#subplots)
- [Scatter & Bar Plot](#scatter-and-bar-plot)

# Introduction to Matplotlib
<img width="800" alt="matplotlib-anatomy-of-a-plot" src="https://user-images.githubusercontent.com/64508435/112073781-b5c42380-8baf-11eb-87db-f4241ea7232a.png">

- `Figure` can contains multiple Subplot
- `Axes 0` and `Axes 1` are `AxesSubplot` stacked together
## Pyplot API vs Object-Oriented API
- In general, try to use the `object-oriented interface` (more flexible) over the `pyplot` interface (i.e: `plt.plot()`

```Python
x = [1,2,3,4]
y = [11,22,33,44]
```
```Python
# Pyplot API
plt.plot(x,y)
```
```Python
# [Recommended] Object-oriented interface 
fig, ax = plt.subplots() #create figure + set of subplots, by default, nrow =1, ncol=1
ax.plot(x,y) #add some data
plt.show()
```
## Matplotlib Workflow
```Python
# 0. Import and get matplotlib ready
%matplotlib inline
import matplotlib.pyplot as plt

# 1. Prepare data
x = [1, 2, 3, 4]
y = [11, 22, 33, 44]

# 2. Setup plot
fig, ax = plt.subplots(figsize=(5,5)) #Figure size = Width & Height of the Plot

# 3. Plot data
ax.plot(x, y)

# 4. Customize plot
ax.set(title="Sample Simple Plot", 
       xlabel="x-axis", 
       ylabel="y-axis")

# 5. Save & Show
fig.savefig("../images/simple-plot.png")
```
# Subplots
- Option #1: to plot multiple subplots in same figure
```Python
# Option 1: Create multiple subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, 
                                             ncols=2, 
                                             figsize=(10, 5))
# Plot data to each axis
ax1.plot(x, x/2);
ax2.scatter(np.random.random(10), np.random.random(10));
ax3.bar(nut_butter_prices.keys(), nut_butter_prices.values());
ax4.hist(np.random.randn(1000));
```
<img width="608" alt="Screenshot 2021-03-23 at 8 52 20 AM" src="https://user-images.githubusercontent.com/64508435/112076325-35a0bc80-8bb5-11eb-980d-3e4cd8a48f10.png">

# Scatter and Bar Plot
