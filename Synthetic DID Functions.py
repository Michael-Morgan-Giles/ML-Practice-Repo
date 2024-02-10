# -*- coding: utf-8 -*-
"""
Synthetic Control Difference in Difference Functions

Most of this code is appropriated from: https://matheusfacure.github.io/python-causality-handbook/25-Synthetic-Diff-in-Diff.html.

All credit for most of the functions goes to the author.

Minor changes have been made so the functions are a tad more generalisable to other datasets.

Their is also a function to generate a randomised dataset based on parameters (including a built in effect post treatment). 

Note, the data is just generated using the rand() function. 

There is scope to switch this to another distribution i.e. normal, beta, gamma, poisson etc.

An example with an actual effect of +2 is included in the code as an example.

"""

#%% Synthetic Control Difference in Differences

# clear memory first
%reset -f

import warnings
warnings.filterwarnings('ignore')

# import necessary packages
import numpy as np
import pandas as pd
from toolz import partial
#import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style
import statsmodels.formula.api as smf
import cvxpy as cp
from joblib import Parallel, delayed
import gc

# Parameters descriptions for the functions:
    # data: the DataFrame containing columns for units, year, the variable of interest, a boolean column for treated and boolean column if observation is after treatment ,
    # outcome_col: column for the variable of interest in data,
    # year_col: column for year in data,
    # unit_col: column for unit in data, 
    # treat_col: column for treatment in data, 
    # post_col column for after treatment in data,
    # unit_w: pd.Series of optomised unit weights, 
    # time_w: pd.Series of optomised time weights

# function to fit time weights
def fit_time_weights(data, outcome_col, year_col, unit_col, treat_col, post_col):
    
    control = data[~data[treat_col]]
    
    # pivot into matrix representation
    y_pre = (control[~control[post_col]].
             pivot_table(values=outcome_col, index=year_col, columns=unit_col))
    
    # group post treatment time periods into an appropriate length vector
    y_post_mean = (control[control[post_col]]
        .groupby(unit_col)
        [outcome_col]
        .mean()
        .values)
    
    # add vector on top of the matrix for intercept calcs
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

# regularisation for unit weights
def calculate_regularization(data, outcome_col, year_col, unit_col, treat_col, post_col):
        n_treated_post = data[data[post_col] & data[treat_col]].shape[0]
        
        first_diff_std = (data[~data[post_col] & ~data[treat_col]]
                      .sort_values(year_col)
                      .groupby(unit_col)
                      [outcome_col]
                      .diff()
                      .std())
    
        return n_treated_post**(1/4) * first_diff_std
    
# fit the unit weights
def fit_unit_weights(data, outcome_col, year_col, unit_col, treat_col, post_col):
    zeta = calculate_regularization(data, outcome_col, year_col, unit_col, treat_col, post_col)
    pre_data = data[~data[post_col]]
    
    # pivot into matrix representation
    y_pre_control = (pre_data[~pre_data[treat_col]]
                     .pivot(year_col, unit_col, outcome_col))
    
    # group treated units by time periods to get vector
    y_pre_treat_mean = (pre_data[pre_data[treat_col]]
                        .groupby(year_col)
                        [outcome_col]
                        .mean())
    
    # sack the matrix and vector
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

# join the weights datasets to the main dataset
def join_weights(data, unit_w, time_w, year_col, unit_col, treat_col, post_col):
    return(
        data.
        set_index([year_col,unit_col])
        .join(time_w)
        .join(unit_w)
        .reset_index()
        .fillna({time_w.name: 1 / len(pd.unique(data[data[post_col]][year_col])),
                 unit_w.name: 1 / len(pd.unique(data[data[treat_col]][unit_col]))})
        .assign(**{"weights": lambda d: (d[time_w.name] * d[unit_w.name]).round(10)})
        .astype({treat_col: int, post_col: int}))

# The synthetic DID function. Returns the models treatment effect parameter. 
def synthetic_diff_in_diff(data, outcome_col, year_col, unit_col, treat_col, post_col):
    # fit unit weights
    unit_weights = fit_unit_weights(data,
                                    outcome_col=outcome_col,
                                    year_col=year_col,
                                    unit_col=unit_col,
                                    treat_col=treat_col,
                                    post_col=post_col)
    
    # fit time weights
    time_weights = fit_time_weights(data,
                                    outcome_col=outcome_col,
                                    year_col=year_col,
                                    unit_col=unit_col,
                                    treat_col=treat_col,
                                    post_col=post_col)    
    
    # join weights
    did_data = join_weights(data, unit_weights, time_weights,
                            year_col=year_col,
                            unit_col=unit_col,
                            treat_col=treat_col,
                            post_col=post_col)
    
    # run sdid
    formula = f"{outcome_col} ~ {post_col}*{treat_col}"
    did_model = smf.wls(formula, data = did_data, weights = did_data['weights']+1e-10).fit()
    
    return did_model.params[f"{post_col}:{treat_col}"]#, did_model.summary()

#%% placebo testing functions

# make placebo data
def make_random_placebo(data, unit_col, treat_col):
    control = data[~data[treat_col]]
    states = control[unit_col].unique()
    placebo_state = np.random.choice(states)
    return control.assign(**{treat_col: control[unit_col] == placebo_state})

# Estimate appropriate standard errors
def estimate_se(data, outcome_col, year_col, unit_col, treat_col, post_col, bootstrap_rounds=400, seed=0, njobs=4):
    np.random.seed(seed=seed)
    
    sdid_fn = partial(synthetic_diff_in_diff,
                      outcome_col = outcome_col, 
                      year_col = year_col, 
                      unit_col = unit_col,
                      treat_col = treat_col,
                      post_col = post_col)
    
    effects = Parallel(n_jobs = njobs)(delayed(sdid_fn)(make_random_placebo(data, unit_col=unit_col, treat_col=treat_col))
                                        for _ in range(bootstrap_rounds))
    
    return np.std(effects, axis = 0)

#%% randomly generate data to implement functions

def generate_synth_data(num_units, num_years, start_year, treated_unit, treatment_year, treatment_effect):
    
    # generate panel
    units = np.repeat(np.arange(1, num_units + 1), num_years)
    years = np.tile(np.arange(1, num_years + 1), num_units)
    years = years + start_year - 1

    np.random.seed(seed)
    variable_data = np.random.rand(num_units * num_years)
    #sd = 10
    #variable_data = np.random.normal(num_units * num_years, sd)
    

    # Assign treated and after_treatment booleans
    treated = np.where(units == treated_unit, True, False)
    after_treatment = np.where(years >= treatment_year, True, False)

    # Apply treatment effect to post treatment treated group
    treatment_effect_array = np.where((units == treated_unit) & (years >= treatment_year), treatment_effect, 0)
    variable_data += treatment_effect_array

    data = pd.DataFrame({
        'unit': units,
        'year': years,
        'variable_of_interest': variable_data,
        'treated': treated,
        'after_treatment': after_treatment
        })
    
    return data

# update parameters as appropriate to generate dataframe
num_units = 5  # i number of units where i = treatment + control
num_years = 10  # years per unit of i
start_year = 2000 # year to start series from (int)
treated_unit = 2  # which unit is the treated variable
treatment_year = start_year + 5 - 1  # Year of intervention (number of years after start of series)
treatment_effect = 2  # actual treatment effect
seed = 42

# use function to generate data
data = generate_synth_data(num_units, num_years, start_year, treated_unit, treatment_year, treatment_effect)

print(data.head()) 
data.info() 

#style.use("fivethirtyeight")
#style.use("seaborn-v0_8")

# graph to compare treatment unit to mean of control group
ax = plt.subplot(1, 1, 1)
(data
 .assign(california = np.where(data["treated"], 'treated_unit', "Control Units"))
 .groupby(["year", "treated"])
 ["variable_of_interest"]
 .mean()
 .reset_index()
 .pivot("year", "treated", "variable_of_interest")
 .plot(ax=ax, figsize=(10,5)))

plt.vlines(x=treatment_year, ymin=0, ymax=treatment_effect+1, linestyle=":", lw=2, label="Treatment")
plt.ylabel("Variable of Interest Trend")
plt.title("Stylised synthetic Control Example")
plt.legend()
plt.show();


#%% running functions through synthetic data

# aggregate effect
synthetic_diff_in_diff(data, 
                       outcome_col="variable_of_interest",
                       year_col="year",
                       unit_col="unit",
                       treat_col="treated",
                       post_col="after_treatment")

gc.collect()

# effect by year
# note this range must be from the treatment year to the max year (or what the year you want to estimate to)

effects = {year : synthetic_diff_in_diff(data[(~data['after_treatment']) | (data['year'] == year)], 
                       outcome_col="variable_of_interest",
                       year_col="year",
                       unit_col="unit",
                       treat_col="treated",
                       post_col="after_treatment")
           for year in range(2004,2010)} 

gc.collect()

effects = pd.Series(effects)

print(effects)

plt.plot(effects);
plt.ylabel("Effect of Variable of Interest")
plt.title("SDID Effect Estimate by Year for Variable of Interest");
plt.show()

#%% run placebo variance estimates and standard errors


# aggregate effects form placebo 
effect = synthetic_diff_in_diff(data,
                       outcome_col="variable_of_interest",
                       year_col="year",
                       unit_col="unit",
                       treat_col="treated",
                       post_col="after_treatment")

gc.collect()

# estimate aggregate standard errors
se = estimate_se(data,
                 outcome_col="variable_of_interest",
                 year_col="year",
                 unit_col="unit",
                 treat_col="treated",
                 post_col="after_treatment")

gc.collect()

print(f"Effect: {effect}")
print(f"Standard Error: {se}")
print(f"95% CI: ({effect-1.96*se}, {effect+1.96*se})")

# effects by year

standard_errors = {year: estimate_se(data[(~data['after_treatment']) | (data['year'] == year)],
                       outcome_col="variable_of_interest",
                       year_col="year",
                       unit_col="unit",
                       treat_col="treated",
                       post_col="after_treatment")
                   for year in range(2004,2010)}

gc.collect()

standard_errors = pd.Series(standard_errors)

plt.figure(figsize=(16,6))
plt.plot(effects,color = "C0")
plt.fill_between(effects.index, effects-1.96*standard_errors, effects+1.96*standard_errors, alpha=0.2,  color="C0")
plt.hlines(y=0,xmin = 2004, xmax = 2009, lw =3, color = 'black',  linestyle=":")
plt.xlabel("Year")
plt.ylabel("Effect of Variable of Interest")
plt.title("SDID Effect Estimate by Year for Variable of Interest with 95% CI");
plt.xticks(rotation=45);


