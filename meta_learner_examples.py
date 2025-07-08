
import pandas as pd
import numpy as np
import gc

# import datasets

#urls = ["https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/refs/heads/master/datasets/IHDP/csv/ihdp_npci_1.csv"]

## initialise list to finalise raw data links
urls = []

## for loop to set up columns
for i in range(1,11):
    urls.append(f"https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/refs/heads/master/datasets/IHDP/csv/ihdp_npci_{i}.csv")

gc.collect()

## initialise dataframe to bind dataframes
df = pd.DataFrame()

## for loop to bind datasets together 
for i in urls:
    df = pd.concat([df, 
               pd.read_csv(i, header=None)])

gc.collect()

# set columns
df.columns = ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"] + [f'x{i}' for i in range(1,26)]

# inspect
df.shape
df.head()
df.describe()

# set up variables to feed into models
y = "y_factual"
T = "treatment"
X = [f'x{i}' for i in range(1,26)]

# load in model
from sklearn.ensemble import GradientBoostingRegressor

params = {'n_estimators' : 400,
          'max_depth' : 4,
          'min_samples_split' : 10,
          'learning_rate' : 0.01,
          'loss' : "squared_error"}






