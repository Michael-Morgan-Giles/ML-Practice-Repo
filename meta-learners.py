# meta-learner estimators
## packages & functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import gc

def elast(df, y, t):
    return np.sum((df[t] - df[t].mean())*(df[y] - df[y].mean())) / np.sum((df[t] - df[t].mean())**2)

def cumulative_gain(df, pred, y, t, min_periods = 30, steps = 100):
    size = df.shape[0]
    ordered_df = df.sort_values(pred, ascending = False).reset_index(drop=True)
    n_rows = list(range(min_periods, size, size // steps)) + [size]
    return np.array([elast(ordered_df.head(rows),y,t) * (rows / size) for rows in n_rows])

# data
test = pd.read_csv("https://raw.githubusercontent.com/matheusfacure/python-causality-handbook/refs/heads/master/causal-inference-for-the-brave-and-true/data/invest_email_rnd.csv")
train = pd.read_csv("https://raw.githubusercontent.com/matheusfacure/python-causality-handbook/refs/heads/master/causal-inference-for-the-brave-and-true/data/invest_email_biased.csv")

# base mod
# the basic idea for future applications, update to the model you want an the code should run i.e. two options for randomforestregressor

### model params
params = {'n_estimators': 400,
                'max_depth': 4,
                'min_samples_split': 10,
                'learning_rate': 0.01,
                'loss': 'squared_error'}

params_rf = {'n_estimators': 400,
                'max_depth': 4,
                'min_samples_split': 10,
                'criterion' : 'squared_error'#
               # 'criterion': 'absolute_error',
                #'criterion': 'friedman_mse'
                }

### base mod for optimisation
#TODO: sort out issue optimising on base_mod object
#base_mod = GradientBoostingRegressor(**params)
#base_mod = RandomForestRegressor(**params_rf)

# set up treatment and covariates
y = "converted"
T = "em1"
X = ["age", "income", "insurance", "invested"]

# S-learner
s_learner = RandomForestRegressor(**params_rf).fit(train[X+[T]], train[y])

s_learner_cate_train = (s_learner.predict(train[X].assign(**{T: 1})) -
                        s_learner.predict(train[X].assign(**{T: 0})))

s_learner_cate_test = test.assign(
    cate=(s_learner.predict(test[X].assign(**{T: 1})) - # predict under treatment
          s_learner.predict(test[X].assign(**{T: 0}))) # predict under control
)

gain_curve_test = cumulative_gain(s_learner_cate_test, "cate", y="converted", t="em1")
gain_curve_train = cumulative_gain(train.assign(cate=s_learner_cate_train), "cate", y="converted", t="em1")
plt.plot(gain_curve_test, color="C0", label="Test")
plt.plot(gain_curve_train, color="C1", label="Train")
plt.plot([0, 100], [0, elast(test, "converted", "em1")], linestyle="--", color="black", label="Baseline")
plt.legend()
plt.title("S-Learner")
plt.show()

# T-learner 
### basic idea behind this regression is to estimate the mean for treatment and control as a function of the features
### \hat \tau (x)_i = M_1(X_i) - M_0(X_i)
np.random.seed(123)
m0 = RandomForestRegressor(**params_rf).fit(train[train[T]==0][X], train[train[T]==0][y])
m1 = RandomForestRegressor(**params_rf).fit(train[train[T]==1][X], train[train[T]==1][y])

# training and test cate
t_learner_cate_train = m1.predict(train[X]) - m0.predict(train[X])
print(t_learner_cate_train)
t_learner_cate_test = m1.predict(test[X]) - m0.predict(test[X])
print(t_learner_cate_test)

# the train and test vectors are the same? 

#plot
gain_curve_test = cumulative_gain(t_learner_cate_test, "cate", y="converted", t="em1")
gain_curve_train = cumulative_gain(train.assign(cate=t_learner_cate_train), "cate", y="converted", t="em1")
plt.plot(gain_curve_test, color="C0", label="Test")
plt.plot(gain_curve_train, color="C1", label="Train")
plt.plot([0, 100], [0, elast(test, "converted", "em1")], linestyle="--", color="black", label="Baseline")
plt.legend()
plt.title("T-Learner")
plt.show()

# X-learner
from sklearn.linear_model import LogisticRegression

ps_mod = LogisticRegression(solver="lbfgs", penalty=None).fit(train[X], train[T])

def logit_ps_predict(df, t, features = X, mod = ps_mod):
    return mod.predict_proba(df[features])[:,t]

# assign weights using previously trained models
d_train = np.where(train[T] == 0,
                   m1.predict(train[X]) - train[y],
                   train[y] - m0.predict(train[X]))

# second stage
mx0 = RandomForestRegressor(**params_rf).fit(train[train[T]==0][X], d_train[train[T]==0])
mx1 = RandomForestRegressor(**params_rf).fit(train[train[T]==1][X], d_train[train[T]==1])

x_cate_train = (logit_ps_predict(train,1)*mx0.predict(train[X]) + logit_ps_predict(train,0)*mx1.predict(train[X]))
print(x_cate_train)

x_cate_test = (logit_ps_predict(test,1)*mx0.predict(test[X]) + logit_ps_predict(test,0)*mx1.predict(test[X]))
print(x_cate_test)

# train and test cate the same again??? (melt)

# again plot cum gain
gain_curve_test = cumulative_gain(x_cate_test, "cate", y="converted", t="em1")
gain_curve_train = cumulative_gain(train.assign(cate=x_cate_train), "cate", y="converted", t="em1")
plt.plot(gain_curve_test, color="C0", label="Test")
plt.plot(gain_curve_train, color="C1", label="Train")
plt.plot([0, 100], [0, elast(test, "converted", "em1")], linestyle="--", color="black", label="Baseline")
plt.legend()
plt.title("X-Learner")
plt.show()

# clear objects
del T, X, y, mx0,mx1,m1,m0,gain_curve_test, gain_curve_train, logit_ps_predict, ps_mod,x_cate_test,x_cate_train, t_learner_cate_test, t_learner_cate_train,s_learner_cate_train, s_learner_cate_train, test, train, elast, cumulative_gain
gc.collect()

# end of script