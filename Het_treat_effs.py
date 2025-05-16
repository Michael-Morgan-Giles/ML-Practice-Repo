# Heterogenous Treatment Effects

### The condoitional ATE (CATE) is the ATE condition on some characteristics specific to the treated unit i.
### Or mathetmatically; $E[Y_1 - Y_0 | X] where X contains the characeristics specific to the treated unit i.

### Below is a simple example implementing a CATE 

# import packages
import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
import statsmodels.api as sm
import seaborn as sns
from matplotlib import pyplot as plt
import gc

# import data 
## update link when necessary
link = "https://raw.githubusercontent.com/matheusfacure/python-causality-handbook/refs/heads/master/causal-inference-for-the-brave-and-true/data/ice_cream_sales_rnd.csv"
## load data and split
df = pd.read_csv(link)

train, test = train_test_split(df, random_state=123)

del df
gc.collect()

# simple model
m1 = smf.ols("sales ~ price + temp + C(weekday) + cost", data=train).fit()
print(m1.summary())

# now the more complex model interacting price with temperature
m2 = smf.ols("sales ~ price*temp + C(weekday) + cost", data=train).fit()
print(m2.summary())
# note inverse relationship for \frac{\del sales}{\del price} where \beta_1 < 0 & \beta_3 > 0

# we can add further interaction terms for all variables - this gives us the CATE
m3 = smf.ols("sales ~ price*cost + price*C(weekday) + price*temp", data=train).fit()
print(m3.summary())
# the interactions between price and days of the week are the conditional part of the cate 
# (where price acts as a continous treatment in this case - company can control this for specific units)

# estimate the sensitivity for each unit (first derivative of sales w.r.t price)
def estimate_sens(model, test_data, T = "price"):
    # note that this function is just \hat y(T + 1) - \hat y(T) where \hat y is the prediction of the model
    return test_data.assign(**{
        "pred_sens" : model.predict(test_data.assign(**{T:test_data[T]+1})) - model.predict(test_data)
    })

model_3_elasticity = estimate_sens(model=m3, test_data=test)

np.random.seed(1)
model_3_elasticity.sample(5)
# `pred_sens` is the estimate of the derivative
# in this case, how much sales would change when inceasing the price by 1

# lets compare to a predictive model
pred_mod = ensemble.GradientBoostingRegressor().fit(train[['temp', 'weekday', 'cost', 'price']], train['sales'])
pred_mod.score(test[['temp', 'weekday', 'cost', 'price']],test['sales'])

elas_comp = model_3_elasticity.assign(
    sens_band = pd.qcut(model_3_elasticity['pred_sens'], 2),
    pred_mod_sales = pred_mod.predict(model_3_elasticity[['temp', 'weekday', 'cost', 'price']]),
    pred_band = pd.qcut(pred_mod.predict(model_3_elasticity[['temp', 'weekday', 'cost', 'price']]), 2) 
)

sns.FacetGrid(elas_comp, col = 'sens_band').map_dataframe(sns.regplot, x = 'price', y = 'sales')
plt.show()

del model_3_elasticity, elas_comp, pred_mod, estimate_sens, m1, m2, m3
gc.collect()

# evaluating models
# extra packages and data
from toolz import curry

link2 = "https://raw.githubusercontent.com/matheusfacure/python-causality-handbook/refs/heads/master/causal-inference-for-the-brave-and-true/data/ice_cream_sales.csv"

## load data and split
df_1 = pd.read_csv(link) # random data
df_2 = pd.read_csv(link2) # non-random data

# estimate models (model 1 is sensitivity model and model 2 is predictive model)
# estimated them above on old data - redo
m_cate = smf.ols("sales ~ price*cost + price*C(weekday) + price*temp", data=df_2).fit()
m_pred = ensemble.GradientBoostingRegressor().fit(df_2[['temp', 'weekday', 'cost', 'price']], df_2['sales'])

print("Train R2: ", r2_score(y_true=df_1['sales'], y_pred=m_pred.predict(df_1[['temp', 'weekday', 'cost', 'price']])))
print("Test R2: ", r2_score(y_true=df_2['sales'], y_pred=m_pred.predict(df_2[['temp', 'weekday', 'cost', 'price']])))

# estimate elasticities (again first derivative as limit formula)
def est_elast(mod, df, inc = 0.01):
    return (mod.predict(df.assign(price = df['price'] + inc)) - mod.predict(df)) / inc

np.random.seed(123)
price_elast_df = df_1.assign(**{
    "cate_m_pred" : est_elast(m_cate, df_1),
    "sales_prediction" : m_pred.predict(df_1[['temp', 'weekday', 'cost', 'price']]),
    "random_sales_prediction" : np.random.uniform(size = df_1.shape[0])
})

# coding up cumulative sensitivity curve
def elasticity(df, y, t):
    return np.sum((df[t] - df[t].mean())*(df[y] - df[y].mean())) / np.sum((df[t] - df[t].mean())**2)

def cumulative_elasticity(df, pred, y, t, min_periods = 30, steps = 100):
    size = df.shape[0]
    ordered_df = df.sort_values(pred, ascending = False).reset_index(drop=True)
    n_rows = list(range(min_periods, size, size // steps)) + [size]
    return np.array([elasticity(ordered_df.head(rows),y,t) for rows in n_rows])

# plot cumulative elasticity
plt.figure(figsize=(10,6))
for m in ['cate_m_pred', 'sales_prediction', 'random_sales_prediction']:
    cum_elast = cumulative_elasticity(price_elast_df, m, 'sales', 'price', min_periods= 100, steps = 100)
    x = np.array(range(len(cum_elast)))
    plt.plot(x/x.max(), cum_elast, label = m)

plt.hlines(elasticity(price_elast_df, 'sales', 'price'), 0, 1, linestyle = "--", color = 'black', label = 'avg elast')
plt.xlabel("% of top elasticity")
plt.ylabel("Cumulative Elasticity")
plt.title("Cumulative Elasticity Curve")
plt.legend()
plt.show()

# cumulative gain curve
def cumulative_elast_gain(df, pred, y, t, min_periods = 30, steps = 100):
    size = df.shape[0]
    ordered_df = df.sort_values(pred, ascending = False).reset_index(drop=True)
    n_rows = list(range(min_periods, size, size // steps)) + [size]
    return np.array([elasticity(ordered_df.head(rows),y,t) * (rows / size) for rows in n_rows])

plt.figure(figsize=(10,6))
for m in ['cate_m_pred', 'sales_prediction', 'random_sales_prediction']:
    cum_elast_gain = cumulative_elast_gain(price_elast_df, m, 'sales', 'price', min_periods= 50, steps = 100)
    x = np.array(range(len(cum_elast_gain)))
    plt.plot(x/x.max(), cum_elast_gain, label = m)

plt.hlines(elasticity(price_elast_df, 'sales', 'price'), 0, 1, linestyle = "--", color = 'black', label = 'avg elast')
plt.xlabel("% of top elasticity")
plt.ylabel("Cumulative Elasticity Gain")
plt.title("Cumulative Elasticity Gain")
plt.legend()
plt.show()

## estimate variance (using typical standard error formula)
def beta_var(df, y, t, z = 1.96):
    n = df.shape[0]
    beta_1 = elasticity(df, y, t)
    beta_0 = df[y].mean() - beta_1*df[t].mean()
    e = df[y] - (beta_0 + beta_1*df[t])
    se = np.sqrt(((1/(n-2))*np.sum(e**2))/np.sum((df[t]-df[t].mean())**2))
    return np.array([beta_1 - z*se, beta_1 +z*se])

def beta_var_cumulative(df, pred, y, t, min_periods = 30, steps = 100):
    size = df.shape[0]
    ordered_df = df.sort_values(pred, ascending = False).reset_index(drop=True)
    n_rows = list(range(min_periods, size, size // steps)) + [size]
    return np.array([beta_var(ordered_df.head(rows),y,t) for rows in n_rows])

plt.figure(figsize=(10,6))
cumu_gain = beta_var_cumulative(price_elast_df, 'cate_m_pred', 'sales', 'price', min_periods= 50, steps = 200)
x = np.array(range(len(cumu_gain)))
plt.plot(x/x.max(), cumu_gain, color = "C0")

plt.hlines(elasticity(price_elast_df, 'sales', 'price'), 0, 1, linestyle = "--", color = 'black', label = 'avg elast')
plt.xlabel("% of top elasticity")
plt.ylabel("Cumulative Elasticity Gain")
plt.title("Cumulative Elasticity Gain")
plt.legend()
plt.show()

# cumulative gain ci
def cumulative_gain_var(df, pred, y, t, min_periods = 30, steps = 100):
    size = df.shape[0]
    ordered_df = df.sort_values(pred, ascending = False).reset_index(drop=True)
    n_rows = list(range(min_periods, size, size // steps)) + [size]
    return np.array([beta_var(ordered_df.head(rows),y,t)*(rows/size) for rows in n_rows])

plt.figure(figsize=(10,6))
cumu_gain = cumulative_gain_var(price_elast_df, 'cate_m_pred', 'sales', 'price', min_periods= 50, steps = 200)
x = np.array(range(len(cumu_gain)))
plt.plot(x/x.max(), cumu_gain, color = "C0")

plt.plot([0,1], [0, elasticity(price_elast_df, 'sales', 'price')], linestyle = "--", color = 'black', label = 'random model')
plt.xlabel("% of top elasticity")
plt.ylabel("Cumulative Elasticity Gain")
plt.title("Cumulative Elasticity Gain")
plt.legend()
plt.show()

# this is all well and great in terms of a causal model validation
# but none of this will work without randomised data ... so not so helpful (sweat)

# end of script