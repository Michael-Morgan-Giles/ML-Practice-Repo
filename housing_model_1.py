# -*- coding: utf-8 -*-
#%% Import Packages
import pandas            as pd
import matplotlib.pyplot as plt
import seaborn           as sns
import numpy             as np
import xgboost           as xgb

from sklearn.preprocessing                  import StandardScaler
from sklearn.metrics                        import mean_squared_error, mean_absolute_error
from sklearn.model_selection                import train_test_split, KFold, RandomizedSearchCV
from sklearn.ensemble                       import RandomForestRegressor
#from sklearn.linear_model                   import Lasso
from sklearn.impute                         import KNNImputer
from sklearn.linear_model                   import Ridge
#from sklearn.linear_model                   import LinearRegression 
#from sklearn.metrics                        import r2_score

# set display format
pd.options.display.float_format = '{:.3f}'.format

# set seed
seed = 42

#%% Import Data

df = pd.read_excel("C:/.../housing_data.xlsx", sheet_name='Data')

df.sort_values(by = 'Date_of_transaction')

#%% cleanse data & EDA
# select appropriate columns
df_clean = df[['Index', 'Sold?', 'Sale_Price', 'Sqm', 
               'Bedrooms', 'Bathrooms', 'Car_Spaces', 
               'Guide_Price', 'Auction?',  'Station', 
               'Shops', 'Plane', 'pa_growth', 'Potential_Rent', 
               'Council_Rates', 'Water_Rates', 'Strata_Levies',
               # added suburb for one hot encoding to model locatioon
               'Suburb']]

# filter for those sold observations
df_clean = df_clean[df_clean['Sold?'] == "Yes"]

# Impute missing values with column median for numerical columns
#df_clean = df_clean.fillna(df_clean.median())

# Check for any NaN or infinite values across rows
#df_clean[df_clean.isin([np.nan, np.inf, -np.inf]).any(axis=1)].count().sum()

# one-hot encode categorical variables
def one_hot_encode_dataframe(df):
    object_cols = df.select_dtypes(include=['object']).columns
    
    df_encoded = pd.get_dummies(df, columns=object_cols)
    
    return df_encoded

encoded_dataframe = one_hot_encode_dataframe(df_clean)
encoded_dataframe.info()

# testing KNN Imputer
imputer = KNNImputer(n_neighbors=5)  

data_filled = imputer.fit_transform(encoded_dataframe)

df_filled = pd.DataFrame(data_filled, columns=encoded_dataframe.columns)

df_filled[df_filled.isin([np.nan, np.inf, -np.inf]).any(axis=1)].count().sum()

# Visualize the confusion matrix using seaborn
#plt.figure(figsize=(8, 6))
#sns.heatmap(encoded_dataframe.corr(), annot=True, cmap="coolwarm")
#plt.title("Correlation Matrix")
#plt.show()

#%% set up variables for models & split into training, test, base & meta sets

X = df_filled[['Sqm', 'Bedrooms', 'Bathrooms', 'Car_Spaces', 'Guide_Price', 
              'Station', 'Shops', 'Plane', 'pa_growth', 'Potential_Rent', 
              'Council_Rates', 'Water_Rates', 'Strata_Levies']]

y = df_filled['Sale_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=seed)

# c is a constant third the length of the dataframe
c = len(y)/3

# set base and meta y dataset
y_meta = y_train[-int(c):]
y_base = y_train[:-int(c)]

# set base and meta X dataset
X_meta = X_train[-int(c):]
X_base = X_train[:-int(c)]


#%% Train base Models

# scale data (mu=0, sd=1)
scaler = StandardScaler()

# transform all X 
X_base = scaler.fit_transform(X_base)
X_meta = scaler.transform(X_meta)
X_test = scaler.transform(X_test)

# set cross validation folds
cv = KFold(n_splits=2, random_state=None, shuffle=False)

# Lasso
#lasso = Lasso(fit_intercept=1, alpha=0.05, max_iter=10000, random_state=seed)

#las_params = {'fit_intercept': [True, False],
#              'alpha': [0.005, 0.01, 0.03, 0.05, 0.07, 0.1],
#              'max_iter': [1000, 2000, 3000],
#              'tol': [0.0001, 0.001, 0.01]}

#gs_las = RandomizedSearchCV(lasso, las_params, cv=cv, scoring='neg_mean_squared_error', 
#                      n_jobs=-1, verbose=1, random_state=seed)

#gs_las.fit(X_base, y_base)

# Random Forest
rf = RandomForestRegressor(n_estimators=400, min_samples_split=3, max_features='sqrt', random_state=seed)

rf_params = {'n_estimators': range(300, 500, 25),
             'min_samples_split': [2, 3, 4, 5, 6, 7],
             'max_features': ['log2', 'sqrt'],  
             'max_depth': [10, 20, 30, 40, 50, None], 
             'min_samples_leaf': [1, 2, 4],  
             'bootstrap': [True, False],  
             'criterion': ['mse', 'mae']}

gs_rf = RandomizedSearchCV(rf, rf_params, cv=cv, scoring = 'neg_mean_squared_error', 
                     n_jobs=-1, verbose=1, random_state=seed)

gs_rf.fit(X_base, y_base)

# XGBoost
xgb = xgb.XGBRegressor(n_estimators=100, gamma=200, eta=0.1, subsample=1, objective='reg:squarederror', random_state=seed)

xgb_params = {
    'n_estimators': range(70, 140, 10),
    'subsample': [0.5, 0.75, 1],
    'eta': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4],
    'gamma': [150],
    #'gamma': range(150, 160, 10),
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_child_weight': [1, 3, 5],
    'colsample_bytree': [0.3, 0.5, 0.7, 1.0],
    'reg_alpha': [0, 0.01, 0.1, 1]
}

gs_xgb = RandomizedSearchCV(xgb, xgb_params, cv=cv, scoring = 'neg_mean_squared_error', n_jobs=-1, verbose=1, random_state=seed)

gs_xgb.fit(X_base, y_base)

#%% train meta model
# Meta models tests
#gs_las.best_params_
gs_rf.best_params_
gs_xgb.best_params_

est_meta = pd.DataFrame(y_meta)

#est_meta['las_pred'] = gs_las.predict(X_meta)
est_meta['rf_pred'] = gs_rf.predict(X_meta)
est_meta['xgb_pred'] = gs_xgb.predict(X_meta)

#plt.figure(figsize=(16,7))
#ax=sns.boxplot(data=est_meta, orient='v', palette = sns.color_palette("deep", 5))
#ax.set(ylim=(0, 500000))
#plt.title('Boxplots of Model Predictions on Meta Set', fontsize=16)
#plt.xlabel('Model')
#plt.ylabel('Sale Price')
#plt.ticklabel_format(style='plain', axis='y');

est_meta = est_meta[[#'las_pred',
                     'rf_pred','xgb_pred']]

# stack model etsimations (Ridge)
param_grid = {
    'alpha': [0.1, 1.0, 10.0],        
    'fit_intercept': [True, False],  
    'normalize': [True, False],      
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],  
    'random_state': [seed]            
}

meta_mod = Ridge()
meta = RandomizedSearchCV(meta_mod, param_grid, cv=5, scoring='neg_mean_squared_error', random_state=seed)

#meta = LinearRegression()
meta.fit(est_meta, y_meta)

#%% Model testing
# testing of meta model against seperate models

reg_dict = {
#    'Lasso Regression': gs_las,
    'Random Forest': gs_rf,
    'XG Boost': gs_xgb
}

def reg_scoring(X, y, meta, reg_dict):
   
    test1_scores = []
    test2_scores = []
    #test3_scores = []
    
    df_pred = pd.DataFrame(columns=reg_dict.keys()) # Columns of DF will accord with reg_dict keys
    
    # Loop through Dictionary items
    for key, reg in reg_dict.items():
        
        pred_y = reg.predict(X)
        df_pred[key] = pd.Series(pred_y).transpose()
        
        # Computing test scores for each model
        test1_scores.append(round(mean_absolute_error(y, pred_y), 3))
        test2_scores.append(round(mean_squared_error(y, pred_y, squared=False), 3))
        #test3_scores.append(round(r2_score(y, pred_y), 3))
                            
    # Generate results DataFrame
    results = pd.DataFrame({'Model': list(reg_dict.keys()), 
                            'Mean Absolute Error': test1_scores,
                            'Root Mean Squared Error': test2_scores#,
                            #'R-Squared': test3_scores
                            })

    # Generate Stack Model's predictions, and test scores
    df_pred['Stack Model'] = meta.predict(df_pred) # Adding 'Stack Model' to DataFrame of predictions
    
    s1 = round(mean_absolute_error(y, df_pred['Stack Model']), 3)
    s2 = round(mean_squared_error(y, df_pred['Stack Model'], squared=False), 3)
    #s3 = round(r2_score(y, df_pred['Stack Model']), 3)
    
    # Add target variable to the DataFrame of predictions
    df_pred['Target'] = y.values.tolist()
        
    # Inserting the Stack scores to the results DataFrame
    row1 = ['Stack Model', s1, s2]#, s3]
    results.loc[len(results)] = row1
    
    return results, df_pred

# make predictions & scores
scores, df_pred = reg_scoring(X_test, y_test, meta, reg_dict)

scores
df_pred

#%% Result Visualisaiton
# graph results

plt.figure(figsize=(16,7))
ax=sns.boxplot(data=df_pred, orient='v', palette = sns.color_palette("deep", 5))
plt.title('Boxplots of Model Predictions on test Set with Target', fontsize=16)
plt.xlabel('Model')
plt.ylabel('Sale Price')
plt.ticklabel_format(style='plain', axis='y')

plt.figure(figsize=(16,7))
ax=sns.histplot(data=df_pred[['Stack Model',#'Lasso Regression', 
                              'Random Forest', 'XG Boost']], kde = True,
             palette = sns.color_palette("deep", 3))
plt.title('Histogram of Model Predictions on test Set with Target', fontsize=16)
plt.xlabel('Model')
plt.ylabel('Sale Price')
plt.vlines(x = df_pred['Target'].mean(), ymin = 0, ymax = 10)
plt.ticklabel_format(style='plain', axis='x')


# melt data to get facets per model
df_melted = df_pred.melt(id_vars=['Target'], 
                         value_vars=['Stack Model', #'Lasso Regression', 
                                     'Random Forest', 'XG Boost'], 
                         var_name='Model', 
                         value_name='Sale Price')

plt.figure(figsize=(16,7))
g = sns.FacetGrid(df_melted, col="Model", col_wrap=2, height=4, aspect=2, sharex=False)
g.map(sns.histplot, "Sale Price", kde=True, palette=sns.color_palette("deep", 4))
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Histograms of Model Predictions on Test Set Target', fontsize=16)
for ax in g.axes.flat:
    ax.axvline(df_pred['Target'].mean(), color='red', linestyle='--', label = "Average Test Price")
    ax.ticklabel_format(style='plain', axis='x')
g.set_axis_labels("Sale Price", "Count")
plt.legend()
plt.show()


#%% New Predictions
# make predictions of new results

# fill in the estimate of below to get an output
X_test_new = pd.DataFrame({
                'Sqm' : [115],      
                'Bedrooms' : [2],       
                'Bathrooms' : [1],       
                'Car Spaces' : [1],       
                'Guide Price' : [950000],  
                'Station' : [1],      
                'Shops' : [0],       
                'Plane' : [0],       
                'pa_growth' : [4286],   
                'Potential_Rent' : [720],
                'Council_Rates' : [353], 
                'Water_Rates' : [178],   
                'Strata_Levies' : [950]
                }
)

X_test_new = scaler.transform(X_test_new)
y_test_new = pd.DataFrame({"Sale_Price": [790000]})
scores_new, df_pred_new = reg_scoring(X_test_new, y_test_new, meta, reg_dict)
df_pred_new


id_to_estimate = 114

X_test_new = df.loc[df['Index'] == id_to_estimate,['Sqm', 'Bedrooms', 'Bathrooms', 'Car_Spaces', 'Guide_Price', 
              'Station', 'Shops', 'Plane', 'pa_growth', 'Potential_Rent', 
              'Council_Rates', 'Water_Rates', 'Strata_Levies']]

X_test_new = X_test_new.fillna(df.median())
X_test_new.isna().sum()
X_test_new = scaler.transform(X_test_new)

# fill in sale price to see how the models performs                          
y_test_new = pd.DataFrame({"Sale_Price": [1250000]})#.values.tolist() # Sale Price
#y_test_new = df_filled.loc[df_filled['Index'] == 121, 'Sale_Price']

scores_new, df_pred_new = reg_scoring(X_test_new, y_test_new, meta, reg_dict)

df_pred_new

for reg in reg_dict.items():
    print(reg.predict(X_test_new))

for key, reg in reg_dict.items():
    pred_y = reg.predict(X_test_new)    
    print(pred_y)

#gs_las.predict(X_test_new)
gs_rf.predict(X_test_new)
gs_xgb.predict(X_test_new)

ids = list(df_clean["Index"].unique())

for i in ids:
    X_test_new = df.loc[df['Index'] == i,['Sqm', 'Bedrooms', 'Bathrooms', 'Car_Spaces', 'Guide_Price', 
              'Station', 'Shops', 'Plane', 'pa_growth', 'Potential_Rent', 
              'Council_Rates', 'Water_Rates', 'Strata_Levies']]

    X_test_new = X_test_new.fillna(df.median())

    X_test_new = scaler.transform(X_test_new)
                     
    y_test_new = pd.DataFrame({"Sale_Price": [790000]})#.values.tolist() # Sale Price

    scores_new, df_pred_new = reg_scoring(X_test_new, y_test_new, meta, reg_dict)
    
    print(df_pred_new)
    

for i in ids:
    X_test_new = df.loc[df['Index'] == i,['Sqm', 'Bedrooms', 'Bathrooms', 'Car_Spaces', 'Guide_Price', 
              'Station', 'Shops', 'Plane', 'pa_growth', 'Potential_Rent', 
              'Council_Rates', 'Water_Rates', 'Strata_Levies']]
    print(X_test_new)


gs_xgb.predict(df.loc[df['Index'] == 2,['Sqm', 'Bedrooms', 'Bathrooms', 'Car_Spaces', 'Guide_Price', 
              'Station', 'Shops', 'Plane', 'pa_growth', 'Potential_Rent', 
              'Council_Rates', 'Water_Rates', 'Strata_Levies']])

