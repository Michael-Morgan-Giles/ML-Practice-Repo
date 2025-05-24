# import necessary packages & custom lift functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
import gc
import statsmodels.formula.api as smf

def elast(df, y, t):
    return np.sum((df[t] - df[t].mean())*(df[y] - df[y].mean())) / np.sum((df[t] - df[t].mean())**2)

def cumulative_gain(df, pred, y, t, min_periods = 30, steps = 100):
    size = df.shape[0]
    ordered_df = df.sort_values(pred, ascending = False).reset_index(drop=True)
    n_rows = list(range(min_periods, size, size // steps)) + [size]
    return np.array([elast(ordered_df.head(rows),y,t) * (rows / size) for rows in n_rows])

# data
test = pd.read_csv("https://raw.githubusercontent.com/matheusfacure/python-causality-handbook/refs/heads/master/causal-inference-for-the-brave-and-true/data/ice_cream_sales_rnd.csv")
train = pd.read_csv("https://raw.githubusercontent.com/matheusfacure/python-causality-handbook/refs/heads/master/causal-inference-for-the-brave-and-true/data/ice_cream_sales.csv")

# Basic idea behind double/debiased ML model is to use the residuals to orthoganilise the data
# this means we predict Y with features X as model M_y
# we predict treatment T with features X as model M_t
# obtain the residuals from each and then regress the residuals of M_Y on M_t

# lets do this first for just the ATE before the CATE

# set y, T, X
y = 'sales'
T = 'price'
X = ['temp', 'weekday', 'cost']

# debiased treatment model (M_T) - note we use cross validation predictions to prevent overfitting
M_t = RandomForestRegressor(n_estimators= 400, max_depth=3)

# this function will return the predictions of treatment based on features X
cross_val_predict(M_t, train[X], train[T])

training_prediction_df = train.assign(price_residuals = train[T] - cross_val_predict(M_t, train[X], train[T], cv = 5) + train[T].mean()) 

# denoise the sales data - again cross validation prediction to avoid overfitting
M_y = RandomForestRegressor(n_estimators=400, max_depth=3)

training_prediction_df = training_prediction_df.assign(sales_residuals = training_prediction_df[y] - cross_val_predict(M_y, train[X], train[y], cv = 5) + training_prediction_df[y].mean())

# ATE model
ATE_model = smf.ols(formula='sales_residuals ~ price_residuals', data=training_prediction_df).fit()
print(ATE_model.summary())

# compare the above coefficient to this one - the above is negative (expected) where as the below is positive (unexpected)
print(smf.ols(formula='sales ~ price', data=training_prediction_df).fit().summary())

# So, the orthoganilisation might be working? ``\_(._.)/``

# Now lets do this but estimate a CATE :)
## The basic steps are the same, but now we just add to the residual model the interactions between the features and treatment residuals.
## Then the cate is basically just the change in sales conditional on X, so:
### M(price = 1, X) - M(price = 0, X)
CATE_model = smf.ols(formula = 'sales_residuals ~ price_residuals * (temp + C(weekday) + cost)', data=training_prediction_df).fit()

# lets estimate the CATE on the test set
cate_test = test.assign(cate = CATE_model.predict(test.assign(price_residuals = 1)) - CATE_model.predict(test.assign(price_residuals = 0)))
cate_train = train.assign(cate = CATE_model.predict(train.assign(price_residuals = 1)) - CATE_model.predict(train.assign(price_residuals = 0)))

gain_curve_test = cumulative_gain(cate_test, "cate", y=y, t=T)
plt.plot(gain_curve_test, color="C0", label="Test")
plt.plot([0, 100], [0, elast(test, y, T)], linestyle="--", color="black", label="Baseline")
plt.legend()
plt.title("R-Learner")
plt.show()

# lets do this same thing but non-parametrically
## the basic idea behind the non-parametric approach is we can optimise directly on the CATE
## This is done with the R-loss function. 
## basically, using the treatment residuals as weights we can target the ratio of the outcome and treatment ratios as the minimum of its square - using any ML model (fancy algebra trick)

# re-estimate the above and don't add mean back (not sure why that wasn't done above ...)
train_pred_non_param_df = train.assign(price_residuals = train[T] - cross_val_predict(M_t, train[X], train[T], cv = 5),
                                       sales_residuals = train[y] - cross_val_predict(M_y, train[X], train[y], cv = 5))

## for this example i'll use a gradient boosted regression
r_learner_model = GradientBoostingRegressor(n_estimators=400, max_depth=3)
r_learner_model = RandomForestRegressor(n_estimators=400, max_depth=3)

# create weights and transform target
weights = train_pred_non_param_df['price_residuals'] ** 2
y_tilde = (train_pred_non_param_df['sales_residuals'] / train_pred_non_param_df['price_residuals'])

r_learner_model.fit(X=train_pred_non_param_df[X], y = y_tilde, sample_weight=weights)

# estimate cate non-parametrically 
non_parametric_cate = test.assign(cate = r_learner_model.predict(test[X]))

# in a weird way, this is basically a local linear estimate of the CATE that in reality is non-lienar. 
# it's kind of like the linear slope (or first derivative) of the non-linear function
# so it's no wise to extrapolate this effect too much - non-linear stuff is too abstract for humans to understand easily

gain_curve_test_non_param = cumulative_gain(non_parametric_cate, "cate", y=y, t=T)
plt.plot(gain_curve_test_non_param, color="C0", label="Non-Parametric")
plt.plot(gain_curve_test, color="C1", label="Parametric")
plt.plot([0, 100], [0, elast(test, y, T)], linestyle="--", color="black", label="Baseline")
plt.legend()
plt.title("R-Learner")
plt.show()



def non_param_r_learner(model, train, test, target_colum, treatment_column, features):
    weights = train[treatment_column] ** 2
    y_tilde = train[target_colum]/train[treatment_column]
    model.fit(train[features], y_tilde, sample_weight = weights)

    return test.assign(cate = model.predict(test[features]))

test_mod = RandomForestRegressor(n_estimators=400, max_depth=3)

non_param_test = non_param_r_learner(model=test_mod, 
                                     train=train_pred_non_param_df, 
                                     test= test, 
                                     target_colum='sales_residuals', 
                                     treatment_column='price_residuals',
                                     features=X)

cumulative_gain(non_param_test, "cate", y=y, t=T)

from sklearn.linear_model import ElasticNet
test_mod2 = ElasticNet(random_state=42)

non_param_test_elast = non_param_r_learner(model=test_mod2, 
                                     train=train_pred_non_param_df, 
                                     test= test, 
                                     target_colum='sales_residuals', 
                                     treatment_column='price_residuals',
                                     features=X)

