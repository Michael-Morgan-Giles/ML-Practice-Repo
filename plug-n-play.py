# plug and play estimators
## packages & functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def elast(df, y, t):
    return np.sum((df[t] - df[t].mean())*(df[y] - df[y].mean())) / np.sum((df[t] - df[t].mean())**2)

def cumulative_gain(df, pred, y, t, min_periods = 30, steps = 100):
    size = df.shape[0]
    ordered_df = df.sort_values(pred, ascending = False).reset_index(drop=True)
    n_rows = list(range(min_periods, size, size // steps)) + [size]
    return np.array([elast(ordered_df.head(rows),y,t) * (rows / size) for rows in n_rows])


## data
link = "https://raw.githubusercontent.com/matheusfacure/python-causality-handbook/refs/heads/master/causal-inference-for-the-brave-and-true/data/invest_email_rnd.csv"

train, test = train_test_split(pd.read_csv(link), test_size = 0.4, random_state=123)

# use mean as propesntiy score to adjust outcome variable
y_star_train = train['converted'] * (train['em1'] - train['em1'].mean()) / (train['em1'].mean() * (1 - train['em1'].mean()))

#import mods
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor

#features
X = ['age', 'income', 'insurance', 'invested']
#alt_ps_scores = LogisticRegression().fit(train[X], train['em1']).predict_proba(train[X])

np.random.seed(123)
cate_learner = GradientBoostingRegressor(n_estimators = 400,
                                         max_depth = 4,
                                         min_samples_split = 10,
                                         learning_rate = 0.01,
                                         loss = "squared_error").fit(train[X], y_star_train)

cate_learner.score(test[X], test['converted'] * (test['em1'] - test['em1'].mean()) / (test['em1'].mean() * (1 - test['em1'].mean())))

# make predictions
test_pred = test.assign(cate = cate_learner.predict(test[X]))
test_pred.head()

# plot
gain_curve_test = cumulative_gain(test_pred, "cate", y="converted", t="em1")
gain_curve_train = cumulative_gain(train.assign(cate=cate_learner.predict(train[X])), "cate", y="converted", t="em1")
plt.plot(gain_curve_test, color="C0", label="Test")
plt.plot(gain_curve_train, color="C1", label="Train")
plt.plot([0, 100], [0, elast(test, "converted", "em1")], linestyle="--", color="black", label="Baseline")
plt.legend()
plt.show()

del gain_curve_test, gain_curve_train, train, test, cate_learner, X, y_star_train
gc.collect()

# continuous case
link = "https://raw.githubusercontent.com/matheusfacure/python-causality-handbook/refs/heads/master/causal-inference-for-the-brave-and-true/data/ice_cream_sales_rnd.csv"

train, test = train_test_split(pd.read_csv(link), test_size = 0.3, random_state=123)

# adjust treatment variable
y_star_continous = (train['price'] - train['price'].mean()*train['sales'] - train['sales'].mean())
X = ["temp", "weekday", "cost"]

np.random.seed(123)
cate_learner = GradientBoostingRegressor(n_estimators = 400,
                                         max_depth = 4,
                                         min_samples_split = 10,
                                         learning_rate = 0.01,
                                         loss = "squared_error").fit(train[X], y_star_continous)

# make predictions
test_pred = test.assign(cate = cate_learner.predict(test[X]))
test_pred.head()

# calc gain
gain_curve_test = cumulative_gain(test_pred, "cate", y="sales", t="price")
gain_curve_train = cumulative_gain(train.assign(cate=cate_learner.predict(train[X])), "cate", y="sales", t="price")

plt.figure(figsize=(10,6))
plt.plot(gain_curve_test, label="Test")
plt.plot(gain_curve_train, label="Train")
plt.plot([0, 100], [0, elast(test, "sales", "price")], linestyle="--", color="black", label="Taseline")
plt.legend()
plt.show()

# endof script