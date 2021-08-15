{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "WOw8yMd1VlnD"
   },
   "source": [
    "# Data Preprocessing "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "NvUGC8QQV6bV"
   },
   "source": [
    "## Importing the libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "id": "wfFEXZC0WS-V"
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "fhYaZ-ENV_c5"
   },
   "source": [
    "## Importing the dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "id": "aqHTg9bxWT_u"
   },
   "outputs": [],
   "source": [
    "dataset = pd.read_csv('Data.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "id": "g1SODqISHgOU"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   Country   Age   Salary Purchased\n",
      "0   France  44.0  72000.0        No\n",
      "1    Spain  27.0  48000.0       Yes\n",
      "2  Germany  30.0  54000.0        No\n",
      "3    Spain  38.0  61000.0        No\n",
      "4  Germany  40.0      NaN       Yes\n",
      "5   France  35.0  58000.0       Yes\n",
      "6    Spain   NaN  52000.0        No\n",
      "7   France  48.0  79000.0       Yes\n",
      "8  Germany  50.0  83000.0        No\n",
      "9   France  37.0  67000.0       Yes\n"
     ]
    }
   ],
   "source": [
    "print(dataset)\n",
    "# print(X)\n",
    "# print(y)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "cYlKy8cAB1TB"
   },
   "source": [
    "### Data Imputation (Missing Data Replacement)\n",
    "Datasets often have missing values and this can cause problems for machine learning algorithms. It is considered good practise to identify and replace missing values in each column of your dateset prior to performing predictive modelling. This method of missing data replacement is referred to as data imputation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      ">0,  missing entries: 0, percentage 0.00\n",
      ">1,  missing entries: 1, percentage 10.00\n",
      ">2,  missing entries: 1, percentage 10.00\n",
      ">3,  missing entries: 0, percentage 0.00\n"
     ]
    }
   ],
   "source": [
    "for i in range(len(dataset.columns)):\n",
    "    missing_data = dataset[dataset.columns[i]].isna().sum()\n",
    "    perc = missing_data / len(dataset) * 100\n",
    "    print('>%d,  missing entries: %d, percentage %.2f' % (i, missing_data, perc))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 288x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize = (4,4)) #is to create a figure object with a given size\n",
    "sns.heatmap(dataset.isna(), cbar=False, cmap='viridis', yticklabels=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "#convert the dataframe into a numpy array by calling values on my dataframe (not necessary), but a habit I prefer\n",
    "X= dataset.iloc[:, :-1].values\n",
    "y = dataset.iloc[:, -1].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "id": "mWaCDTjUB6S4"
   },
   "outputs": [],
   "source": [
    "from sklearn.impute import SimpleImputer\n",
    "\n",
    "#Create an instance of Class SimpleImputer: np.nan is the empty value in the dataset\n",
    "imputer = SimpleImputer(missing_values=np.nan, strategy='mean')\n",
    "\n",
    "#Replace missing value from numerical Col 1 'Age', Col 2 'Salary'\n",
    "#fit on the dataset to calculate the statistic for each column\n",
    "imputer.fit(X[:, 1:3]) \n",
    "\n",
    "#The fit imputer is then applied to the dataset \n",
    "# to create a copy of the dataset with all the missing values \n",
    "# for each column replaced with the calculated mean statistic.\n",
    "#transform will replace & return the new updated columns\n",
    "X[:, 1:3] = imputer.transform(X[:, 1:3])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "086YTfEsE1zS",
    "outputId": "b6be8fa2-a121-4258-af45-e1d99f287310"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[['France' 44.0 72000.0]\n",
      " ['Spain' 27.0 48000.0]\n",
      " ['Germany' 30.0 54000.0]\n",
      " ['Spain' 38.0 61000.0]\n",
      " ['Germany' 40.0 63777.77777777778]\n",
      " ['France' 35.0 58000.0]\n",
      " ['Spain' 38.77777777777778 52000.0]\n",
      " ['France' 48.0 79000.0]\n",
      " ['Germany' 50.0 83000.0]\n",
      " ['France' 37.0 67000.0]]\n"
     ]
    }
   ],
   "source": [
    "print(X)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "DSbKT73UTLjq"
   },
   "source": [
    "## Encode Categorical Data\n",
    "#### Encode Independent variable (X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "id": "GE_afh9zTr5o"
   },
   "outputs": [],
   "source": [
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.preprocessing import OneHotEncoder\n",
    "#transformers: specify what kind of transformation, and which cols\n",
    "#Tuple ('encoder' encoding transformation, instance of Class OneHotEncoder, [col to transform])\n",
    "#remainder =\"passthrough\" > to keep the cols which not be transformed. Otherwise, the remaining cols will not be included \n",
    "ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])] , remainder=\"passthrough\" )\n",
    "#fit and transform with input = X\n",
    "#np.array: need to convert output of fit_transform() from matrix to np.array\n",
    "X = np.array(ct.fit_transform(X))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "Lq80jsTIY_n0",
    "outputId": "fef87bf8-f64c-470d-ecc1-6762fcd29524"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1.0 0.0 0.0 44.0 72000.0]\n",
      " [0.0 0.0 1.0 27.0 48000.0]\n",
      " [0.0 1.0 0.0 30.0 54000.0]\n",
      " [0.0 0.0 1.0 38.0 61000.0]\n",
      " [0.0 1.0 0.0 40.0 63777.77777777778]\n",
      " [1.0 0.0 0.0 35.0 58000.0]\n",
      " [0.0 0.0 1.0 38.77777777777778 52000.0]\n",
      " [1.0 0.0 0.0 48.0 79000.0]\n",
      " [0.0 1.0 0.0 50.0 83000.0]\n",
      " [1.0 0.0 0.0 37.0 67000.0]]\n"
     ]
    }
   ],
   "source": [
    "print(X)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "YXXF5M3xZUA1"
   },
   "source": [
    "#### Encode Dependent Variable"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "id": "O054hXMtZTx_"
   },
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import LabelEncoder\n",
    "le = LabelEncoder()\n",
    "#output of fit_transform of Label Encoder is already a Numpy Array\n",
    "y = le.fit_transform(y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "VmdJJTofZwyi",
    "outputId": "97adaef3-356a-47b0-8cd3-70dc15a74464"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0 1 0 0 1 1 0 1 0 1]\n"
     ]
    }
   ],
   "source": [
    "print(y)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "3abSxRqvWEIB"
   },
   "source": [
    "## Splitting the dataset (X = data, y = output) into the Training set and Test set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "id": "hm48sif-WWsh"
   },
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "Nw7mYJpeDi6s",
    "outputId": "125be395-50a3-4590-848b-403a05ec4e75"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0.0 0.0 1.0 38.77777777777778 52000.0]\n",
      " [0.0 1.0 0.0 40.0 63777.77777777778]\n",
      " [1.0 0.0 0.0 44.0 72000.0]\n",
      " [0.0 0.0 1.0 38.0 61000.0]\n",
      " [0.0 0.0 1.0 27.0 48000.0]\n",
      " [1.0 0.0 0.0 48.0 79000.0]\n",
      " [0.0 1.0 0.0 50.0 83000.0]\n",
      " [1.0 0.0 0.0 35.0 58000.0]]\n"
     ]
    }
   ],
   "source": [
    "print(X_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "318jeGZqDjhz",
    "outputId": "b3446b91-7633-47fa-897e-afb5b163b3ea"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0.0 1.0 0.0 30.0 54000.0]\n",
      " [1.0 0.0 0.0 37.0 67000.0]]\n"
     ]
    }
   ],
   "source": [
    "print(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "P-WZ1_yDDjoN",
    "outputId": "f2a8efa2-a1e7-477e-8bc4-6834b531cc51"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0 1 0 0 1 1 0 1]\n"
     ]
    }
   ],
   "source": [
    "print(y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "4C5Z1CgWDjyc",
    "outputId": "f97e7747-5a13-4acf-8f29-bc2debaf65aa"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0 1]\n"
     ]
    }
   ],
   "source": [
    "print(y_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "5S5t--lxERd0"
   },
   "source": [
    "## Feature Scaling\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "id": "ZlRjMJwbEUcP"
   },
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "sc = StandardScaler()\n",
    "X_train[:,3:] = sc.fit_transform(X_train[:,3:])\n",
    "#only use Transform to use the SAME scaler as the Training Set\n",
    "X_test[:,3:] = sc.transform(X_test[:,3:])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "c7dUZrlgS2Vy",
    "outputId": "e4b06995-b32b-40bc-a57b-43b2a41b19ad"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0.0 0.0 1.0 -0.19159184384578545 -1.0781259408412425]\n",
      " [0.0 1.0 0.0 -0.014117293757057777 -0.07013167641635372]\n",
      " [1.0 0.0 0.0 0.566708506533324 0.633562432710455]\n",
      " [0.0 0.0 1.0 -0.30453019390224867 -0.30786617274297867]\n",
      " [0.0 0.0 1.0 -1.9018011447007988 -1.420463615551582]\n",
      " [1.0 0.0 0.0 1.1475343068237058 1.232653363453549]\n",
      " [0.0 1.0 0.0 1.4379472069688968 1.5749910381638885]\n",
      " [1.0 0.0 0.0 -0.7401495441200351 -0.5646194287757332]]\n"
     ]
    }
   ],
   "source": [
    "print(X_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "T270Isd8T10Y",
    "outputId": "66e4d916-cdff-479b-809b-3be3cf313003",
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0.0 1.0 0.0 -1.4661817944830124 -0.9069571034860727]\n",
      " [1.0 0.0 0.0 -0.44973664397484414 0.2056403393225306]]\n"
     ]
    }
   ],
   "source": [
    "print(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "\n",
    "## 5. Training Machine Learning Model\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "## Models from Scikit-Learn: Search \"scikit learn model map\"\n",
    "from sklearn.linear_model import LogisticRegression\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LogisticRegression()"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "logistic_clf = LogisticRegression()\n",
    "logistic_clf.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 5.1. Evaluate the model\n",
    "\n",
    "Now we've made some predictions, we can start to use some more Scikit-Learn methods to figure out how good our model is. \n",
    "\n",
    "Each model or estimator has a built-in score method. This method compares how well the model was able to learn the patterns between the features and labels. In other words, it returns how accurate your model is."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.75"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Evaluate the model on the training set\n",
    "logistic_clf.score(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.5"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Evaluate the model on the test set\n",
    "logistic_clf.score(X_test, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 1])"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_preds = logistic_clf.predict(X_test)\n",
    "y_preds"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1])"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0])"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Predict with a single input\n",
    "logistic_clf.predict([[0.0, 0.0, 1.0, -0.19159184384578545, -1.0781259408412425]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "collapsed_sections": [],
   "name": "data_preprocessing_template.ipynb",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
