
# Matplotlib Cheat Sheet
# Table of contents
- [Table of contents](#table-of-contents)
- [Introduction to Matplotlib](#introduction-to-matplotlib)
  - [Plotting from an IPython notebook](#plotting-from-an-ipython-notebook)
  - [Matplotlib Two Interfaces: MATLAB-style & Object-Oriented Interfaces](#matplotlib-two-interfaces)
  - [Matplotlib Workflow](#matplotlib-workflow) 
- [Subplots](#subplots)
- [Scatter, Bar & Histogram Plot](#scatter-bar-and-histogram-plot)

# Introduction to Matplotlib
- Matplotlib is a multi-platform data visualization library built on NumPy arrays, and designed to work with the broader SciPy stack
- Newer tools like `ggplot` and `ggvis` in the R language, along with web visualization toolkits based on `D3js` and `HTML5 canvas`, often make Matplotlib feel clunky and old-fashioned
- Hence, nowadays, cleaner, more modern APIs, for example, `Seaborn`, `ggpy`, `HoloViews`, `Altai`, has been developed to drive Matplotlib
```Python
import matplotlib.pyplot as plt
```
- The `plt` interface is what we will use most often
#### Setting Styles
```Python
# See the different styles avail
plt.style.available
# Set Style
plt.style.use('seaborn-whitegrid')
```
## Plotting from an IPython notebook
- `%matplotlib notebook` will lead to **interactive** plots embedded within the notebook
- `%matplotlib inline`   will lead to **static images** of your plot embedded in the notebook

<img width="800" alt="matplotlib-anatomy-of-a-plot" src="https://user-images.githubusercontent.com/64508435/112073781-b5c42380-8baf-11eb-87db-f4241ea7232a.png">

- `Figure` can contains multiple Subplot
- `Axes 0` and `Axes 1` are `AxesSubplot` stacked together
## Matplotlib Two Interfaces
### Pyplot API vs Object-Oriented API
* Quickly  &#8594; use Pyplot Method
* Advanced &#8594; use Object-Oriented Method
- In general, try to use the `object-oriented interface` (more flexible) over the `pyplot` interface (i.e: `plt.plot()`)

```Python
x = [1,2,3,4]
y = [11,22,33,44]
```
- **MATLAB-style or PyPlot API**: Matplotlib was originally written as a Python alternative for MATLAB users, and much of its syntax reflects that fact
```Python
# Pyplot API
plt.plot(x,y, color='blue')

plt.title("A Sine Curve") #in OO, use the ax.set() method to set all these properties at once
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.xlim([1,3])
plt.ylim([20,])
```
- **Object-oriented**: plotting functions are methods of explicit `Figure` and `Axes` objects.
```Python
# [Recommended] Object-oriented interface 
fig, ax = plt.subplots() #create figure + set of subplots, by default, nrow =1, ncol=1
ax.plot(x,y) #add some data
plt.show()
```
##### Matplotlib Gotchas

While most `plt` functions translate directly to `ax` methods (such as plt.plot() → ax.plot(), plt.legend() → ax.legend(), etc.), this is not the case for all commands. In particular, functions to set limits, labels, and titles are slightly modified. For transitioning between MATLAB-style functions and object-oriented methods, make the following changes:
  - `plt.xlabel()` → `ax.set_xlabel()`
  - `plt.ylabel()` → `ax.set_ylabel()`
  - `plt.xlim()` → `ax.set_xlim()`
  - `plt.ylim()` → `ax.set_ylim()`
  - `plt.title()` → `ax.set_title()`
In the object-oriented interface to plotting, rather than calling these functions individually, it is often more convenient to use the ax.set() 

```Python
ax.set(xlim=(0, 10), ylim=(-2, 2),
       xlabel='x', ylabel='sin(x)',
       title='A Simple Plot');
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
       ylabel="y-axis", 
       xlim=(0, 10), ylim=(-2, 2))

# 5. Save & Show
fig.savefig("../images/simple-plot.png")
```

[(Back to top)](#table-of-contents)

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

[(Back to top)](#table-of-contents)

# Scatter Bar and Histogram Plot
## Scatter
```Python
#<--- Method 1: Pytlot --->: 
df.plot(kind = 'scatter',
             x = 'age',
             y = 'chol',
             c = 'target', #c = color the dot based on over_50['target'] columns
             figsize=(10,6));
```
```Python
#<--- Method 2: OO --->: 
## OO Method from Scratch
fig, ax = plt.subplots(figsize=(10,6))

## Plot the data
scatter = ax.scatter(x=over_50["age"],
                     y=over_50["chol"],
                     c=over_50["target"]);
# Customize the plot
ax.set(title="Heart Disease and Cholesterol Levels",
       xlabel="Age",
       ylabel="Cholesterol");
# Add a legend
ax.legend(*scatter.legend_elements(), title="target"); # * to unpack all the values of Title="target"

#Add a horizontal line
ax.axhline(over_50["chol"].mean(), linestyle = "--");
```

<img src="https://user-images.githubusercontent.com/64508435/112741395-15f40480-8fb8-11eb-991f-d326cf9399b2.png"/>

## Bar 
* Vertical 
* Horizontal
```Python
#<--- Method 1: Pytlot --->: 
df.plot.bar();
```
```Python
#<--- Method 2: OO --->: 
fig, ax = plt.subplots()
ax.bar(x, y)
ax.set(title="Dan's Nut Butter Store", ylabel="Price ($)");
```
## Histogram
```Python
# Create Histogram of Age to see the distribution of age

heart_disease["age"].plot.hist(bins=10);
```
[(Back to top)](#table-of-contents)
