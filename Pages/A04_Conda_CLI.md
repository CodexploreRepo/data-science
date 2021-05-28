
| Usage              |  Command                           | Description     |
| -------------------| ---------------------------------- | ----------------|
| Create             |`conda create --prefix ./env pandas numpy matplotlib scikit-learn jupyter`| Create Conda env  & install packages |
| List Env           | `conda env list`                   |  Listdown env currently activated  |
| Activate           | `conda activate ./env`             |  Activate Conda virtual env |
| Install package    | `conda install jupyter`            | |
| Update package     | `conda update scikit-learn=0.22`   | Can specify the version also |
| List Installed Package | `conda list`||
| Un-install Package     | `conda uninstall python scikit-learn`| To uninstall packages to re-install with the Latest version|
| Open Jupyter Notebook  | `jupyter notebook`||



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
| Show Function Description | `Shift + Tab`|<img width="707" alt="Screenshot 2021-03-18 at 8 09 57 AM" src="https://user-images.githubusercontent.com/64508435/111554485-694ca280-87c1-11eb-9fb2-3fc946bc332b.png">|
| How to install a conda package into the current env from Jupyter's Notebook|`import sys`<br>`!conda install --yes --prefix {sys.prefix} seaborn`||

### Jupyter Magic Function
| Function           |  Command                           | Description     |
| -------------------| ---------------------------------- | ----------------|
| Matplotlib         | `%matplotlib inline`               | will make your plot outputs appear and be stored within the notebook. |
