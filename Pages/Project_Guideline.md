# End-to-End Machine Learning Project Guideline
<img width="1396" alt="Screenshot 2021-06-24 at 23 01 18" src="https://user-images.githubusercontent.com/64508435/123286072-2b89b380-d540-11eb-9cd9-206687ccf80a.png">

## Table of contents
- [1. Project Environment Setup](#1-project-environment-setup)
  - [1.1 Setup Conda Env](#11-setup-conda-env) 



## 1. Project Environment Setup 
<img width="1396" alt="Screenshot 2021-06-24 at 23 00 56" src="https://user-images.githubusercontent.com/64508435/123286872-ddc17b00-d540-11eb-9fc7-117ead30cfa4.png">

### 1.1. Setup Conda Env
#### 1.1.1. Create Conda Env from Stratch
`conda create --prefix ./env pandas numpy matplotlib scikit-learn jupyter`
#### 1.1.2. Create Conda Env from a base env
- **Step 1**: Go to Base Env folder and export the base conda env to `environment.yml` file
  - *Note*: open  `environment.yml` file by `vim environment.yml` to open the file &#8594; To exit Vim: `press ESC then ; then q to exit`
```Python
conda env list #to list down current env
conda activate /Users/quannguyen/Data_Science/Conda/env #Activate the base conda env
conda env export > environment.yml #Export base conda env to environment.yml file
conda deactivate #de-activate env once done
```
- **Step 2**: Go to current project folder and create the env based on `environment.yml` file
```python
conda env create --prefix ./env -f environment.yml
```
[(Back to top)](#table-of-contents)
