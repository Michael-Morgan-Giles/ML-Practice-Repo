# import packages
import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import seaborn as sns
from matplotlib import pyplot as plt
import gc

# import data 

## update link when necessary
link = "https://raw.githubusercontent.com/matheusfacure/python-causality-handbook/refs/heads/master/causal-inference-for-the-brave-and-true/data/customer_transactions.csv"
link_features = "https://raw.githubusercontent.com/matheusfacure/python-causality-handbook/refs/heads/master/causal-inference-for-the-brave-and-true/data/customer_features.csv"

df = pd.read_csv(link)

# split test / train
train, test = train_test_split(pd.read_csv(link_features).merge(df[['customer_id']].assign(net_value = df.drop(columns='customer_id').sum(axis=1)), on = "customer_id"), 
                               test_size=0.3,
                               random_state=13)

#TODO: install sci-kit learn

del df
gc.collect()

# nice pretty graph
plt.figure(figsize=(12,6))
np.random.seed(123)
#sns.barplot(data=train.assign(income_quant = pd.qcut(train['income'], q = 20)), x = "income_quant", y = "net_value")
sns.barplot(train, x = 'region', y = "net_value")
plt.show()

# goupby
regions_to_net = train.groupby("region")['net_value'].agg(['mean','count','std'])

regions_to_net = regions_to_net.assign(
    lower_bound=regions_to_net['mean'] - 1.96*regions_to_net['std']/(regions_to_net['count']**0.5)
)

regions_to_net = regions_to_net['mean'].to_dict()

def encode(df): 
    return df.replace({"region": regions_to_net})


# Model train

params_dict = {'n_estimators': 400,
                'max_depth': 4,
                'min_samples_split': 10,
                'learning_rate': 0.01,
                'loss': 'squared_error'}

X = ["region", "income", 'age']
y = 'net_value'

np.random.seed(123)

mod = ensemble.GradientBoostingRegressor(**params_dict)

mod.fit(train[X].pipe(encode), train[y])

training_predict = (train[X].pipe(encode).assign(pred = mod.predict(train[X].pipe(encode))))

print("Train R2: ", r2_score(y_true=train[y], y_pred=training_predict["pred"]))
print("Test R2: ", r2_score(y_true=test[y], y_pred=mod.predict(test[X].pipe(encode))))

# build in decision boundary
## save model prediction
mod_data = test
mod_data['prediction'] = mod.predict(mod_data[X].pipe(encode))

## if E[Y|X]>0 business else no business

def mod_bin(pred,bins):
    bands = pd.qcut(pred,q=bins, retbins=True)[1]

    def bin_func(pred):
        return np.digitize(pred, bands)
    return bin_func

bin_fun = mod_bin(training_predict['pred'], 20)

band_df = mod_data.assign(band = bin_fun(mod_data['prediction']))

plt.figure(figsize=(12,6))
sns.barplot(band_df,x = 'band', y = "net_value")
plt.show()

band_df['invest'] = np.where(band_df['band'] > 15, "Invest","Don't Invest")

band_df.groupby("invest")['net_value'].mean()

# remove everything
del train, test, regions_to_net, training_predict, mod, X, y, params_dict, encode, mod_data, band_df, bin_fun, mod_bin
gc.collect()

# end of script