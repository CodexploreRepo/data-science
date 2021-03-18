
| Usage              |  Command                           | Description     |
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

## Jupyter Notebook
| Usage              |  Command                           | Description     |
| -------------------| ---------------------------------- | ----------------|
| Run Cell           |`Shift + Enter`| Create Conda env  & install packages |
| Switch to Markdown | Exit Edit Mode `ESC` > press `m` | |
| List down Function Description | `Shift + Tab`|<img width="707" alt="Screenshot 2021-03-18 at 8 09 57 AM" src="https://user-images.githubusercontent.com/64508435/111554485-694ca280-87c1-11eb-9fb2-3fc946bc332b.png">|
