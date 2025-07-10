
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
from sklearn.model_selection import train_test_split

## train test split
train, test = train_test_split(df, test_size = 0.2, random_state=123)

## model params (gradient boosted reg)
params = {'n_estimators' : 400,
          'max_depth' : 4,
          'min_samples_split' : 10,
          'learning_rate' : 0.01,
          'loss' : "squared_error"}

# S-learner
np.random.seed(42)
s_mod = GradientBoostingRegressor(**params).fit(train[X+[T]], train[y])

test_s = test.assign(cate = s_mod.predict(test[X].assign(**{T : 1})) - s_mod.predict(test[X].assign(**{T : 0})))

test_s['cate'].describe()
test_s['cate'].hist()

# T-learner
np.random.seed(42)
m0 = GradientBoostingRegressor(**params).fit(train[train[T] == 0][X], train[train[T] == 0][y])
m1 = GradientBoostingRegressor(**params).fit(train[train[T] == 1][X], train[train[T] == 1][y])

test_t = test.assign(cate = m1.predict(test[X]) - m0.predict(test[X]))

test_t['cate'].describe()
test_t['cate'].hist()

# X-learner
from sklearn.linear_model import LogisticRegression

## propensity model
p_mod = LogisticRegression(solver="lbfgs", penalty=None).fit(train[X], train[T])

## propensity helper function
def logit_ps_predict(df, t, features = X, mod = p_mod):
    return mod.predict_proba(df[features])[:,t]

## impute cates (using T-learner)
train_impute_cate = np.where(train[T] == 0,
                             m1.predict(train[X]) - train[y], # imputed cate tau_0
                             train[y] - m0.predict(train[X])) # imputed cate tau_1

## stage two mods
np.random.seed(42)
mx0 = GradientBoostingRegressor(**params).fit(train[train[T] == 0][X], train_impute_cate[train[T] == 0])
mx1 = GradientBoostingRegressor(**params).fit(train[train[T] == 1][X], train_impute_cate[train[T] == 1])

# use prop scores to weight model preds
test_x = test.assign(cate = logit_ps_predict(test,1)*mx0.predict(test[X]) + logit_ps_predict(test,0)*mx1.predict(test[X]))
print(test_x)

test_x['cate'].describe()
test_x['cate'].hist()

# R-learner (param)
from sklearn.model_selection import cross_val_predict
import statsmodels.formula.api as smf ## for easier OLS estimation

## debaised nuisance function (T) & cross validate prediction to avoid overfitting on any one section
m_t = GradientBoostingRegressor(**params)

df_train_r = train.assign(treatment_residuals = train[T] - cross_val_predict(m_t, train[X], train[T], cv = 5)) 

## denoise nuisance function (y) & cross val pred
m_y = GradientBoostingRegressor(**params)

df_train_r = df_train_r.assign(outcome_residuals = train[y] - cross_val_predict(m_y, train[X], train[y], cv = 5))

## train cate mod (this case use an ols estimator)
param_cate_mod = smf.ols(formula = f'outcome_residuals ~ treatment_residuals * ({" + ".join(X)})', data=df_train_r).fit()

test_r_param = test.assign(cate = param_cate_mod.predict(test.assign(treatment_residuals = 1)) - param_cate_mod.predict(test.assign(treatment_residuals = 0)))

test_r_param['cate'].describe()
test_r_param['cate'].hist()

# R-learner (non-param)
## train new dataset
df_train_non_param_r = train.assign(outcome_residuals = train[T] - cross_val_predict(m_t, train[X], train[T], cv = 5),
                                       treatment_residuals = train[y] - cross_val_predict(m_y, train[X], train[y], cv = 5))

## mod to optomise non-param function 
non_param_mod = GradientBoostingRegressor(**params)

## set up weights and outcome target
weights = df_train_non_param_r['treatment_residuals'] ** 2 # squared orthogonal treatment indicator
y_tilde = (df_train_non_param_r['outcome_residuals'] / df_train_non_param_r['treatment_residuals']) # ratio of orthogonal outcome and treatment indicator

## optimise using transformed targets and weights
non_param_mod.fit(X = train[X], y = y_tilde, sample_weight=weights)

test_r_non_param = test.assign(cate = non_param_mod.predict(test[X]))

test_r_non_param['cate'].describe()
test_r_non_param['cate'].hist()

# how do we know which one is better? 
## model uplit


# feature selection 
from ml_causal import FilterSelect

filter_method = FilterSelect()

# F Filter with order 1
method = 'F'
f_imp = filter_method.get_importance(df, X_names, y_name, method,
                      treatment_group = 'treatment1')
f_imp.head()

# F Filter with order 2
method = 'F'
f_imp = filter_method.get_importance(df, X_names, y_name, method,
                      treatment_group = 'treatment1', order=2)
f_imp.head()

# LR Filter with order 1
method = 'LR'
lr_imp = filter_method.get_importance(df, X_names, y_name, method,
                      treatment_group = 'treatment1', order = 1)
lr_imp.head()

method = 'KL'
kl_imp = filter_method.get_importance(df, X_names, y_name, method,
                      treatment_group = 'treatment1',
                      n_bins=10)
kl_imp.head()



