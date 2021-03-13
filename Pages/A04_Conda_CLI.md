
| Usage              |  Command                           | Remark  |
| -------------------| ---------------------------------- | ----------------|
| Create             |`conda create --prefix ./env pandas numpy matplotlib scikit-learn jupyter`| Create Conda env  & install packages |
| Activate           | `conda activate ./env`             |  Activate Conda virtual env |
| Install package    | `conda install jupyter`            | |
| List               | `conda env list`                   |  Listdown env currently activated  |
| Open Jupyter Notebook | `jupyter notebook`||


## Sharing Conda Environment
- Share a `.yml` (pronounced YAM-L) file of your Conda environment
- `.yml` is basically a text file with instructions to tell Conda how to set up an environment.
  - Step 1: Export `.yml` file: 
    - `conda env export --prefix {Path to env folder} > environment.yml`
  - Step 2: New PC, create  an environment called `env_from_file` from `environment.yml`:
    -   `conda env create --file environment.yml --name env_from_file`
