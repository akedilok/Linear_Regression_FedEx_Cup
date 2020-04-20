import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools import add_constant
import patsy
from patsy import dmatrices

from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

def df_maker(dictionary):
    '''
    Takes input of triple nested dictionaries, extracts content into dataframe
    '''

    category_list = list(dictionary.keys())
    #cat_count = len(category_list)
    #print(cat_count)

    master_df = pd.DataFrame()
    master_df['Player'] = None
    master_df['Season'] = None

    for cat in category_list:
        #print(cat)
    
        master_df[str(cat)] = None
        #display(master_df)
    
        season_list = list(dictionary[cat].keys())
        season_count = len(season_list)
        #print(season_list)
    
        for season in season_list:
            #print(season)
        
            df = pd.DataFrame()
        
            player_list = list(dictionary[cat][season].keys())
            #print(player_list)
        
            player_count = len(player_list)
            #print(player_count)
        
            stat_list = list(dictionary[cat][season].values())
            stat_count = len(stat_list)
            #print(stat_list)
        
            df['Player'] = player_list
            df['Season'] = season
            df[str(cat)] = stat_list
            #display(df)
        
            master_df = master_df.append(df,sort=False).reset_index(drop=True)
            #display(master_df)
            #master_df = master_df.groupby(['Player','Season']).sum().reset_index()
            #print(f'ABOVE master table updated for {season} {cat}')
    print(f'F@&$ yea! Your data is below!')
    return master_df

########################################################################################################

def defaultdict_to_df(list_of_dicts):
    '''
    Takes pulled PGA data stored in triple nested defaultdict structure, extracts and loads into df.
    Takes input of list of dictionaries to extract data from. 
    '''
    counter = 0
    overall_data_store = []

    for dict_ in list_of_dicts:
        for cat_key in dict_.keys():
            for year in dict_[cat_key][0].keys():
                if len(dict_[cat_key][0][year]) > 0:
                    for player_tup in dict_[cat_key][0][year][0]:
    #                 print(len(player_tup))
                        row = {}
                        row['year'] = year
                        row['player'] = player_tup[0]
                        row['metric'] = cat_key
                        row['value'] = player_tup[1]
                        overall_data_store.append(row)
    return overall_data_store
        
########################################################################################################
    
def parse_to_inch(distance):
    '''
    Converts x' y" data to inches
    '''
    dist = distance.split('\'')
    feet = dist[0]
    inch = dist[1].strip().replace('"','')
    dist_inch = int(feet)*12 + int(inch)
    return dist_inch

########################################################################################################

def series_to_inches(column_name,df):
    '''
    Converts x' y" data to inches for all columns of df specified in column_name list
    Uses parse_to_inch function. 
    '''
    for name in column_name:
        df[name] = df[name].apply(lambda x: parse_to_inch(x))
    return df.info()

########################################################################################################

def to_float(column_name,df):
    '''
    Converts column of df to float from string. 
    '''
    for name in column_name:
        df[name] = df[name].apply(lambda x: float(str(x).replace(',','').strip()))
    return df.info()

########################################################################################################

def heat_map(df):  
    '''
    Creates heatmap using df data.
    '''  
    corr = df.corr()
    #display(corr.describe)

    plt.figure(figsize = (80,80))
    ax = sns.heatmap(corr,
                     vmin = -1, 
                     vmax = 1, 
                     center = 0,
                     cmap = sns.diverging_palette(20,220,n=200),
                     square = True,
                     annot = True)
    ax.set_xticklabels(ax.get_xticklabels(),
                       rotation = 45,
                       horizontalalignment = 'right');

########################################################################################################

def patsy_input_str(df,y_col_name):
    '''
    Creates a string in Y ~ X1 + X2 + X3 ... format for input into patsy.dmatrices
    in order to create feature matrix (X) and target vector (y).
    '''
    col_name_list = df.columns
    string_for_patsy = ''
    for col_name in col_name_list:
        if col_name == y_col_name:
            y_string = str(y_col_name)+' ~ '
            string_for_patsy = y_string + string_for_patsy
        else:
            string_for_patsy = string_for_patsy+str(col_name)+' + '
    if string_for_patsy.strip()[-1] == '+':
        last_letter_pos = len(string_for_patsy)-2
        string_for_patsy = string_for_patsy[:int(last_letter_pos)].strip()
    return string_for_patsy

########################################################################################################

def ols_fit_train(y_array,df,col_list):
    '''
    Takes df and string name of y column name in df, uses patsy_input_str to create string and inputs into feature matrix.
    Outputs OLS fit summary.
    '''
    input_string = patsy_input_str(df,y_col_name)
    # Create your feature matrix (X) and target vector (y)

    y, X = patsy.dmatrices(input_string, data=df, return_type="dataframe")

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=10)
    
    # Create your model
    model = sm.OLS(y_array, add_constant(df.loc[:,col_list]))

    # Fit your model to your training set
    fit = model.fit()

    # Print summary statistics of the model's performance
    return fit.summary()

########################################################################################################

def skl_lin_reg(X_tr,X_val,y_tr,y_val):
    #Regression with Sklearn LinearRegression()

    lr = LinearRegression()

    X_train_lr = X_tr
    X_val_lr = X_val

    # Choose the response variable(s)
    y_train_lr = y_tr
    y_val_lr = y_val

    # Fit the model to the training data
    lr.fit(X_train_lr, y_train_lr)

    # Print out the R^2 for the model against the training set
    print(f'Training set R^2 is: {lr.score(X_train_lr,y_train_lr)}')
    print(f'Validation set R^2 is: {lr.score(X_val_lr,y_val_lr)}')

########################################################################################################

def reg_models(X_train, y_train, alpha, kf):
    r2 = defaultdict(list)
    rmse = defaultdict(list)
    mae = defaultdict(list)

    for train, test in kf.split(X_train,y_train):
        
        X_tr, X_val = X_train.iloc[train],X_train.iloc[test]
        y_tr, y_val = y_train.iloc[train],y_train.iloc[test]
        
    #     mean_cols = [x for x in X_tr.columns if x[:4] == 'mean']
    #     sum_cols = [x for x in X_tr.columns if x[:3] == 'sum']
        
        #scale
        scaler = StandardScaler()
        scaler.fit(X_tr)
        
        X_tr_scaled = scaler.transform(X_tr)
        X_val_scaled = scaler.transform(X_val)
        
        #Instantiate
        lr = LinearRegression()
        l1 = Lasso()
        l2 = Ridge()
        
        #Fit
        lr.fit(X_tr,y_tr)
        l1.fit(X_tr_scaled,y_tr)
        l2.fit(X_tr_scaled,y_tr)
        
        #Predict
        lr_preds = lr.predict(X_val)
        l1_preds = l1.predict(X_val_scaled)
        l2_preds = l2.predict(X_val_scaled)
        
        #R2
        r2['lr'].append(lr.score(X_val,y_val))
        r2['l1'].append(l1.score(X_val_scaled,y_val))
        r2['l2'].append(l2.score(X_val_scaled,y_val))
        
        #RMSE
        rmse['lr'].append(np.sqrt(metrics.mean_squared_error(y_val,lr_preds)))
        rmse['l1'].append(np.sqrt(metrics.mean_squared_error(y_val,l1_preds)))
        rmse['l2'].append(np.sqrt(metrics.mean_squared_error(y_val,l2_preds)))
        
        #MAE
        mae['lr'].append(metrics.mean_absolute_error(y_val,lr_preds))
        mae['l1'].append(metrics.mean_absolute_error(y_val,l1_preds))
        mae['l2'].append(metrics.mean_absolute_error(y_val,l2_preds))

    for model in ['lr','l1','l2']:
        print(f'{model}: \n R2:{np.mean(r2[model])} RMSE:{np.mean(rmse[model])} MAE:{np.mean(mae[model])}')