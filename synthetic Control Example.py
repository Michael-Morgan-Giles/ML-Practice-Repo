# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 12:58:52 2024

@author: Micha

From: https://matheusfacure.github.io/python-causality-handbook/15-Synthetic-Control.html
&
From: https://matheusfacure.github.io/python-causality-handbook/25-Synthetic-Diff-in-Diff.html#diff-in-diff-revisited
"""

## example of synthetic control

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from matplotlib import style
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

#%matplotlib inline

pd.set_option("display.max_columns", 6)
style.use("fivethirtyeight")

cigar = (pd.read_csv("C:/Users/Micha/Documents/5. Misc/13. Misc Python Scripts/Synthetic Control/smoking.csv").drop(columns=["lnincome","beer", "age15to24"]))

cigar.query("california").head()
cigar[cigar['california']==True].head()
cigar.describe()

#%%
### plot california against other states
ax = plt.subplot(1, 1, 1)

(cigar
 .assign(california = np.where(cigar["california"], "California", "Other States"))
 .groupby(["year", "california"])
 ["cigsale"]
 .mean()
 .reset_index()
 .pivot("year", "california", "cigsale")
 .plot(ax=ax, figsize=(10,5)))

plt.vlines(x=1988, ymin=40, ymax=140, linestyle=":", lw=2, label="Proposition 99")
plt.ylabel("Cigarette Sales Trend")
plt.title("Gap in per-capita cigarette sales (in packs)")
plt.legend();

 #%% setting up Synthetic Control Using OLS to calcualte the weights
 
features = ['cigsale', 'retprice']
 
inverted = cigar[cigar['after_treatment']==False].pivot(index = 'state', columns = 'year')[features].T

inverted.head()

### split california and other donor states
y = inverted[3].values
X = inverted.drop(columns = 3).values

# use OLS to calculate weights (do not calcualte with intercept)
# then use weights to create a weighted average for synth control
weights_lr = LinearRegression(fit_intercept=False).fit(X,y).coef_
weights_lr.round(3)

calif_synth_lr = cigar[cigar['california']==False].pivot(index = 'year', columns = 'state')['cigsale'].values.dot(weights_lr)

# graph synth

plt.figure(figsize=(10,6))
plt.plot(cigar[cigar['california']==True]['year'],cigar[cigar['california']==True]['cigsale'], label = 'california')
plt.plot(cigar[cigar['california']==True]['year'],calif_synth_lr, label = 'Synthetic Control')
plt.vlines(x=1988, ymin=40, ymax=140, linestyle=":", lw=2, label="Proposition 99")
plt.ylabel("Gap in per-capita cigarette sales (in packs)")
plt.legend()

# graph of effect

d = {'years': cigar[cigar['california']==True]['year'], 
     "cigsales in California": cigar[cigar['california']==True]['cigsale'],
     "Synthetic Control": calif_synth_lr}
test_df = pd.DataFrame(d)

test_df['calif_synth_lr_treatment_effect'] = test_df["cigsales in California"]-test_df["Synthetic Control"]

plt.figure(figsize=(10,6))
plt.plot(cigar[cigar['california']==True]['year'],cigar[cigar['california']==True]['cigsale'], label = 'california')
plt.plot(cigar[cigar['california']==True]['year'],calif_synth_lr, label = 'Synthetic Control')
plt.plot(test_df['years'],test_df['calif_synth_lr_treatment_effect'], label = 'treatment effect')
plt.vlines(x=1988, ymin=-75, ymax=140, linestyle=":", lw=2, label="Proposition 99")
plt.ylabel("Gap in per-capita cigarette sales (in packs)")
plt.legend()


#%% optomising the weights via interpolation. So the weights will be a convex combination of the donor pool. 
# We can do this by ensuring the weights are positive and sum to one.  
#from typing import List
#from operator import add
#from toolz import reduce, partial
from toolz import  partial
from scipy.optimize import fmin_slsqp

def loss_w(W, X, y) -> float:
    return np.sqrt(np.mean((y - X.dot(W))**2))

def get_w(X, y):
    
    w_start = [1/X.shape[1]]*X.shape[1]

    weights = fmin_slsqp(partial(loss_w, X=X, y=y),
                         np.array(w_start),
                         f_eqcons=lambda x: np.sum(x) - 1,
                         bounds=[(0.0, 1.0)]*len(w_start),
                         disp=False)
    return weights
    
calif_weights = get_w(X, y)
print("Sum:", calif_weights.sum())
np.round(calif_weights, 4)

calif_synth = cigar[cigar['california']==False].pivot(index = 'year', columns = 'state')['cigsale'].values.dot(calif_weights)


# now graph this new synth 
plt.figure(figsize=(10,6))
plt.plot(cigar[cigar['california']==True]['year'],cigar[cigar['california']==True]['cigsale'], label = 'california')
plt.plot(cigar[cigar['california']==True]['year'],calif_synth, label = 'Synthetic Control')
plt.vlines(x=1988, ymin=40, ymax=140, linestyle=":", lw=2, label="Proposition 99")
plt.ylabel("Gap in per-capita cigarette sales (in packs)")
plt.legend()

# plot the treatment effect
plt.figure(figsize=(10,6))
plt.plot(cigar[cigar['california']==True]['year'],cigar[cigar['california']==True]['cigsale']- calif_synth, label="California Synth")
plt.vlines(x=1988, ymin=-30, ymax=7, linestyle=":", lw=2, label="Proposition 99")
plt.hlines(y=0, xmin=1970, xmax=2000, lw=2)
plt.title("State - Synthetic Across Time")
plt.ylabel("Gap in per-capita cigarette sales (in packs)")
plt.legend();

#%% awesome we have a treatment effect, but how do we know if it's statistically significant? 

# set up a function that runs placebos by setting false interventions to the donor pool

from joblib import Parallel, delayed

def synthetic_control(state: int, data: pd.DataFrame) -> np.array:
    
    features = ["cigsale", "retprice"]
    
    inverted = (data.query("~after_treatment")
                .pivot(index='state', columns="year")[features]
                .T)
    
    y = inverted[state].values # treated
    X = inverted.drop(columns=state).values # donor pool

    weights = get_w(X, y)
    synthetic = (data.query(f"~(state=={state})")
                 .pivot(index='year', columns="state")["cigsale"]
                 .values.dot(weights))

    return (data
            .query(f"state=={state}")[["state", "year", "cigsale", "after_treatment"]]
            .assign(synthetic=synthetic))

control_pool = cigar["state"].unique()

parallel_fn = delayed(partial(synthetic_control, data=cigar))

synthetic_states = Parallel(n_jobs=8)(parallel_fn(state) for state in control_pool)
    
# to compare the placebo results, lets plot against the other data
plt.figure(figsize = (16,6))
for state in synthetic_states:
    plt.plot(state['year'], state['cigsale'] - state['synthetic'], color = "C5", alpha = 0.4)

plt.plot(cigar[cigar['california']==True]['year'],cigar[cigar['california']==True]['cigsale']- calif_synth, label="California Synth")
plt.vlines(x=1988,ymin=-50, ymax=120,linestyle=":", lw = 2, label = 'proposition 99')
plt.hlines(y=0,xmin = 1970, xmax = 2000, lw =3)
plt.ylabel("Gap in per-capita cigarette sales (in packs)")
plt.title("State - Synthetic Across Time")
plt.legend();

# for units from the donor pool with no good fit from a convex combination, it is best to remove them.
# we can do this in a data driven way
def pre_treatment_error(state):
    pre_treat_error = (state[state['after_treatment']==False]['cigsale'] - state[state['after_treatment']==False]['synthetic'])**2
    return pre_treat_error.mean()

plt.figure(figsize = (16,6))
for state in synthetic_states:
    
    if pre_treatment_error(state)<80:
        plt.plot(state['year'], state['cigsale'] - state['synthetic'], color = "C5", alpha = 0.4)

plt.plot(cigar[cigar['california']==True]['year'],cigar[cigar['california']==True]['cigsale']- calif_synth, label="California Synth")
plt.vlines(x=1988,ymin=-50, ymax=120,linestyle=":", lw = 2, label = 'proposition 99')
plt.hlines(y=0,xmin = 1970, xmax = 2000, lw =3)
plt.ylabel("Gap in per-capita cigarette sales (in packs)")
plt.title("State - Synthetic Across Time")
plt.legend();

# derive a p-value. Count how many times the effects from the placebo were below the effects of california.

calif_numb = 3

effects = [state[state['year']==2000].iloc[0]['cigsale'] - state[state['year']==2000].iloc[0]['synthetic']
            for state in synthetic_states
            if pre_treatment_error(state)<80]

calif_effect  = cigar[(cigar['year']==2000)&(cigar['california']==True)].iloc[0]['cigsale'] - calif_synth[-1]

print("California Treatment Effect for the Year 2000:", calif_effect)
np.array(effects)

np.mean(np.array(effects) < calif_effect)

# we can plot the distribution effects in a histogram as well
_, bins, _ = plt.hist(effects, bins=20, color="C5", alpha=0.4);
plt.hist([calif_effect], bins=bins, color="C0", label="California")
plt.ylabel("Frquency")
plt.title("Distribution of Effects")
plt.legend();


#%% Synthetic Difference - in - Differences

import numpy as np
import pandas as pd
#from toolz import curry, partial
#import seaborn as sns
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf
import cvxpy as cp
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings('ignore')

from matplotlib import style
style.use("ggplot")

pd.set_option('display.max_columns', 10)


data = (pd.read_csv("C:/Users/Micha/Documents/5. Misc/13. Misc Python Scripts/Synthetic Control/smoking.csv")[["state", "year", "cigsale", "california", "after_treatment"]]
        .rename(columns={"california": "treated"})
        .replace({"state": {3: "california"}}))

# function to fit time weights
def fit_time_weights(data, outcome_col, year_col, state_col, treat_col, post_col):
    
    control = data.query(f"~{treat_col}")
    
    # pivot into matrix representation
    y_pre = (control
             .query(f"~{post_col}")
             .pivot(year_col,state_col,outcome_col))
    
    # group post treatment time periods into appropriate length vector
    y_post_mean = (control
        .query(f"{post_col}")
        .groupby(state_col)
        [outcome_col]
        .mean()
        .values)
    
    # add the N_co vector on top of the matrix for intercept calcs
    X = np.concatenate([np.ones((1, y_pre.shape[1])), y_pre.values], axis=0)
    
    # estimate w_t
    w = cp.Variable(X.shape[0])
    objective = cp.Minimize(cp.sum_squares(w@X - y_post_mean))
    constraints = [cp.sum(w[1:]) == 1, w[1:] >=0]
    problem = cp.Problem(objective, constraints)
    problem.solve(verbose = False)
    
    return pd.Series(w.value[1:],
                     name = 'time_weight',
                     index = y_pre.index)

time_weights = fit_time_weights(data, 
                                outcome_col = 'cigsale',
                                year_col = 'year', 
                                state_col = 'state',
                                treat_col = 'treated',
                                post_col = 'after_treatment')

time_weights.round(3).tail()

# complicate regularisation process for unit weights
def calculate_regularization(data, outcome_col, year_col, state_col, treat_col, post_col):
        n_treated_post = data.query(post_col).query(treat_col).shape[0]
    
        first_diff_std = (data
                      .query(f"~{post_col}")
                      .query(f"~{treat_col}")
                      .sort_values(year_col)
                      .groupby(state_col)
                      [outcome_col]
                      .diff()
                      .std())
    
        return n_treated_post**(1/4) * first_diff_std


# fit the unit weights
def fit_unit_weights(data, outcome_col, year_col, state_col, treat_col, post_col):
    zeta = calculate_regularization(data, outcome_col, year_col, state_col, treat_col, post_col)
    pre_data = data.query(f"~{post_col}")
    
    # pivot the data to the (T_pre, N_co) matrix representation
    y_pre_control = (pre_data
                     .query(f"~{treat_col}")
                     .pivot(year_col, state_col, outcome_col))
    
    # group treated units by time periods to have a (T_pre, 1) vector.
    y_pre_treat_mean = (pre_data
                        .query(f"{treat_col}")
                        .groupby(year_col)
                        [outcome_col]
                        .mean())
    
    # add a (T_pre, 1) column to the begining of the (T_pre, N_co) matrix to serve as intercept
    T_pre = y_pre_control.shape[0]
    X = np.concatenate([np.ones((T_pre, 1)), y_pre_control.values], axis=1) 
    
    # estimate unit weights with L_2 penalty loss
    w = cp.Variable(X.shape[1])
    objective = cp.Minimize(cp.sum_squares(X@w - y_pre_treat_mean.values) + T_pre*zeta**2 * cp.sum_squares(w[1:]))
    constraints = [cp.sum(w[1:]) == 1, w[1:] >= 0]
    
    problem = cp.Problem(objective, constraints)
    problem.solve(verbose=False)
    
    return pd.Series(w.value[1:],
                     name = 'unit_weights',
                     index = y_pre_control.columns)

unit_weights = fit_unit_weights(data,
                                outcome_col="cigsale",
                                year_col="year",
                                state_col="state",
                                treat_col="treated",
                                post_col="after_treatment")

unit_weights.round(3).head()

# join the weights function

def join_weights(data, unit_w, time_w, year_col, state_col, treat_col, post_col):
    return(
        data.
        set_index([year_col,state_col])
        .join(time_w)
        .join(unit_w)
        .reset_index()
        .fillna({time_w.name: 1 / len(pd.unique(data.query(f"{post_col}")[year_col])),
                 unit_w.name: 1 / len(pd.unique(data.query(f"{treat_col}")[state_col]))})
        .assign(**{"weights": lambda d: (d[time_w.name] * d[unit_w.name]).round(10)})
        .astype({treat_col: int, post_col: int}))

did_data = join_weights(data, unit_weights, time_weights,
                        year_col="year",
                        state_col="state",
                        treat_col="treated",
                        post_col="after_treatment")

did_data.head()

## estimate synth did model with weight least squares
did_model = smf.wls("cigsale ~ after_treatment*treated",
                    data = did_data,
                    weights = did_data['weights']+1e-10).fit()

print(did_model.summary())

#%% accounting for time effect heterogeneity and staggered adoption of treatment

def synthetic_diff_in_diff(data, outcome_col, year_col, state_col, treat_col, post_col):
    # fit unit weights
    unit_weights = fit_unit_weights(data,
                                    outcome_col=outcome_col,
                                    year_col=year_col,
                                    state_col=state_col,
                                    treat_col=treat_col,
                                    post_col=post_col)
    
    # fit time weights
    time_weights = fit_time_weights(data,
                                    outcome_col=outcome_col,
                                    year_col=year_col,
                                    state_col=state_col,
                                    treat_col=treat_col,
                                    post_col=post_col)    
    
    # join weights
    did_data = join_weights(data, unit_weights, time_weights,
                            year_col=year_col,
                            state_col=state_col,
                            treat_col=treat_col,
                            post_col=post_col)
    
    # run did
    formula = f"{outcome_col} ~ {post_col}*{treat_col}"
    did_model = smf.wls(formula, data = did_data, weights = did_data['weights']+1e-10).fit()
    
    return did_model.params[f"{post_col}:{treat_col}"]

synthetic_diff_in_diff(data, 
                       outcome_col="cigsale",
                       year_col="year",
                       state_col="state",
                       treat_col="treated",
                       post_col="after_treatment")

## calc effects across years

effects = {year : synthetic_diff_in_diff(data.query(f"~after_treatment | (year == {year})"), 
                       outcome_col="cigsale",
                       year_col="year",
                       state_col="state",
                       treat_col="treated",
                       post_col="after_treatment")
           for year in range(1989,2001)}

effects = pd.Series(effects)

plt.plot(effects);
plt.ylabel("Effect in Cigarette Sales")
plt.title("SDID Effect Estimate by Year");


#%% placebo variance estimate

# make placebo data
def make_random_placebo(data, state_col, treat_col):
    control = data.query(f"~{treat_col}")
    states = control[state_col].unique()
    placebo_state = np.random.choice(states)
    return control.assign(**{treat_col: control[state_col] == placebo_state})

np.random.seed(1)

placebo_data = make_random_placebo(data,state_col = "state", treat_col = "treated")

placebo_data[placebo_data['treated']==True].tail()


# standard error estimates function
def estimate_se(data, outcome_col, year_col, state_col, treat_col, post_col, bootstrap_rounds=400, seed=0, njobs=4):
    np.random.seed(seed=seed)
    
    sdid_fn = partial(synthetic_diff_in_diff,
                      outcome_col = outcome_col, 
                      year_col = year_col, 
                      state_col = state_col,
                      treat_col = treat_col,
                      post_col = post_col)
    
    effects = Parallel(n_jobs = njobs)(delayed(sdid_fn)(make_random_placebo(data, state_col=state_col, treat_col=treat_col))
                                        for _ in range(bootstrap_rounds))
    
    return np.std(effects, axis = 0)

effect = synthetic_diff_in_diff(data,
                                outcome_col="cigsale",
                                year_col="year",
                                state_col="state",
                                treat_col="treated",
                                post_col="after_treatment")


se = estimate_se(data,
                 outcome_col="cigsale",
                 year_col="year",
                 state_col="state",
                 treat_col="treated",
                 post_col="after_treatment")

print(f"Effect: {effect}")
print(f"Standard Error: {se}")
print(f"95% CI: ({effect-1.96*se}, {effect+1.96*se})")

# graph the outcomes
standard_errors = {year: estimate_se(data.query(f"~after_treatment|(year=={year})"),
                 outcome_col="cigsale",
                 year_col="year",
                 state_col="state",
                 treat_col="treated",
                 post_col="after_treatment")
                   for year in range(1989,2001)}

standard_errors = pd.Series(standard_errors)

plt.figure(figsize=(16,6))
plt.plot(effects,color = "C0")
plt.fill_between(effects.index, effects-1.96*standard_errors, effects+1.96*standard_errors, alpha=0.2,  color="C0")
plt.hlines(y=0,xmin = 1989, xmax = 2000, lw =3, color = 'black')
plt.ylabel("Effect in Cigarette Sales")
plt.xlabel("Year")
plt.title("Synthetic DiD 90% Conf. Interval")
plt.xticks(rotation=45);




