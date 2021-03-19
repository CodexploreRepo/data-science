# Statistics
# Table of contents
- [Standard Deviation & Variance](standard-deviation-and-variance)



# Standard Deviation and Variance

## Standard Deviation
- Standard Deviation is a measure of how spread out numbers are.
```Python
# Standard deviation = a measure of how spread out a group of numbers is from the mean
np.std(a2)
# Standar deviation = Square Root of Variance
np.sqrt(np.var(a2))
```

## Variance
- The average of the squared differences from the Mean.
```Python
# Varainace = measure of the average degree to which each number is different to the mean
# Higher variance = wider range of numbers
# Lower variance = lower range of numbers
np.var(a2)
```

### Example:
![image](https://user-images.githubusercontent.com/64508435/111798728-4d521980-8905-11eb-890a-afe682a02c3e.png)
- The heights (at the shoulders) are: 600mm, 470mm, 170mm, 430mm and 300mm
- Mean = (600 + 470 + 170 + 430 + 300)/5 = 394mm

![image](https://user-images.githubusercontent.com/64508435/111799090-a8840c00-8905-11eb-8064-6890d95abca4.png)
- `Variance` = 21704
- `Standard Deviation` = sqrt(variance) = 147 mm

![Screenshot 2021-03-19 at 10 51 56 PM](https://user-images.githubusercontent.com/64508435/111799145-b6399180-8905-11eb-990b-2c72b9067520.png)

- we can show which heights are within one Standard Deviation (147mm) of the Mean:
- **Standard Deviation we have a "standard" way of knowing what is normal**, and what is extra large or extra small

![image](https://user-images.githubusercontent.com/64508435/111799454-fd278700-8905-11eb-98c1-f9866d34f27b.png)
- Credit:  [Math is Fun](https://www.mathsisfun.com/data/standard-deviation.html)
