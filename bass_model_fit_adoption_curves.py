import numpy as np
import pandas as pd
import random
import os
from os.path import join, exists
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import patches
from copy import deepcopy
from utils import *
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
from scipy.optimize import least_squares

"""
This script is for fitting the Bass model curve for the solar adoption curve in each census block
group. For each block group, there will be a set of estimated Bass model parameters, include d, p,
q, m.
"""
price_trend = np.array([10.0, 10.1, 9.6, 9.2, 7.9, 7.0, 5.9, 5.1, 4.8, 4.7, 4.4])
log_price_trend = np.log(price_trend / price_trend[0])

def get_cum_installations(coefs, t):
    """Given Bass model coefficients, and a time array, return the array of cumulative values."""
    p, q, m, d = coefs
    y = (1 - np.exp(-(p + q) * (t - d))) / (1 + q / p * np.exp(-(p + q) * (t - d)))
    return m * y * (t >= d) + 0 * (t < d)

def get_residuals_cum(coefs, t, y_true):
    """
    Given Bass model coefficients, and a time array, return resididual between true cumulative
    values and predicted cumulative values (predicted by Bass model).
    """
    y_pred = get_cum_installations(coefs, t)
    return y_pred - y_true

def get_cum_installations_GBM(coefs, t):
    """
    Given Generalized Bass model coefficients, and a time array, return the array of 
    cumulative values.
    """
    p, q, m, d, beta = coefs
    y = (1 - np.exp(-(p + q) * (t - d + beta * log_price_trend))) / (1 + q / p * np.exp(-(p + q) * (t - d + beta * log_price_trend)))
    return m * y * (t >= d) + 0 * (t < d)

def get_residuals_cum_GBM(coefs, t, y_true):
    """
    Given Generalized Bass model coefficients, and a time array, return resididual between true
    cumulative values and predicted cumulative values (predicted by Bass model).
    """
    y_pred = get_cum_installations_GBM(coefs, t)
    return y_pred - y_true

def get_best_fit_NLS(y_arr, dy_arr, initial_p=0.01, initial_q=0.5, initial_m=None, upper_m=np.inf):
    """
    Given the arrays of cumulative installations in previous year and new installations in current year,
    fit the Bass model curve using Non-Linear Least Square (NLS) with initial values and bounds.
    Different initial values of d (onset of adoption) will be tried to get its best initial value.
    Args:
        y_arr: the array of cumulative installations in previous year.
        dy_arr: the array of new installations in present year.
        initial_p: initial value of p.
        initial_q: initial value of q.
        initial_m: initial value of m.
        upper_m: the upper bound of m.
    Return:
        min_rmse: the minimum RMSE of fitting.
        best_params: the best set of Bass model parameters (p, q, m, d).
        best_y_pred: the prediction of cumulative installations under the best parameters.
        best_dy_pred: the prediction of yearly new installations under the best parameters.
        best_onset_idx: the best initial value of d that will yield the minimum RMSE.
    """
    y_true = np.array(list(y_arr[1:]) + [y_arr[-1] + dy_arr[-1]])
    if initial_m is None:
        initial_m = y_true[-1]
    min_rmse = np.inf
    best_params = None
    best_y_pred = None
    best_onset_idx = None
    for onset_idx in range(-5, 11):
        ls_model = least_squares(
            get_residuals_cum,
            x0=np.array([initial_p, initial_q, initial_m, onset_idx]), 
            jac='cs',
            bounds=([0, 0, 0, -8], [1, np.inf, upper_m, np.inf]),
            args=(np.arange(0, 11), y_true),
            method='trf',
        )
        y_pred = get_cum_installations(ls_model.x, np.arange(0, 11))
        rmse = np.sqrt(np.sum((y_pred - y_true) ** 2))
        p, q, m, d = ls_model.x
        if rmse < min_rmse and p > 0 and q > 0 and m > 0:
            min_rmse = rmse
            best_params = ls_model.x
            best_y_pred = y_pred
            best_onset_idx = onset_idx
    best_dy_pred = np.concatenate([[best_y_pred[0]], best_y_pred[1:] - best_y_pred[:-1]])
    return min_rmse, best_params, best_y_pred, best_dy_pred, best_onset_idx

def get_best_fit_NLS_GBM(y_arr, dy_arr, initial_p=0.01, initial_q=0.5, initial_m=None, initial_beta=-0.4):
    """
    Given the arrays of cumulative installations in previous year and new installations in current year,
    fit the Generalized Bass model (GBM) curve using Non-Linear Least Square (NLS) with initial values and bounds.
    Different initial values of d (onset of adoption) will be tried to get its best initial value.
    Args:
        y_arr: the array of cumulative installations in previous year.
        dy_arr: the array of new installations in present year.
        initial_p: initial value of p.
        initial_q: initial value of q.
        initial_m: initial value of m.
        upper_m: the upper bound of m.
        unitial_beta: the initial value of beta.
    Return:
        min_rmse: the minimum RMSE of fitting.
        best_params: the best set of Bass model parameters (p, q, m, d, beta).
        best_y_pred: the prediction of cumulative installations under the best parameters.
        best_dy_pred: the prediction of yearly new installations under the best parameters.
        best_onset_idx: the best initial value of d that will yield the minimum RMSE.
    """
    y_true = np.array(list(y_arr[1:]) + [y_arr[-1] + dy_arr[-1]])
    if initial_m is None:
        initial_m = y_true[-1]
    min_rmse = np.inf
    best_params = None
    best_y_pred = None
    best_onset_idx = None
    for onset_idx in range(-5, 11):
        ls_model = least_squares(
            get_residuals_cum_GBM,
            x0=np.array([initial_p, initial_q, initial_m, onset_idx, initial_beta]), 
            jac='cs',
            bounds=([0, 0, 0, -np.inf, -np.inf], [1, np.inf, np.inf, np.inf, 0]),
            args=(np.arange(0, 11), y_true),
            method='trf',
        )
        y_pred = get_cum_installations_GBM(ls_model.x, np.arange(0, 11))
        rmse = np.sqrt(np.sum((y_pred - y_true) ** 2))
        p, q, m, d, beta = ls_model.x
        if rmse < min_rmse and p > 0 and q > 0 and m > 0:
            min_rmse = rmse
            best_params = ls_model.x
            best_y_pred = y_pred
            best_onset_idx = onset_idx
    best_dy_pred = np.concatenate([[best_y_pred[0]], best_y_pred[1:] - best_y_pred[:-1]])
    return min_rmse, best_params, best_y_pred, best_dy_pred, best_onset_idx

# 1. Load and process data
bg = pd.read_csv('results/merged_bg.csv')

bg = bg[['blockgroup_FIPS', 'year', 'num_of_installations', 'num_of_buildings_lt600']]

bg['tract_FIPS'] = bg['blockgroup_FIPS'] // 10
bg['county_FIPS'] = bg['blockgroup_FIPS'] // 10000000
bg['state_FIPS'] = bg['blockgroup_FIPS'] // 10000000000

df = bg.sort_values(['year', 'blockgroup_FIPS'])

cumulative_pv_count_dict = {}
cumulative_pv_count = np.array(df[df['year'] == 2005]['num_of_installations'])
cumulative_pv_count_dict[2005] = cumulative_pv_count.copy()
for year in range(2006, 2018):
    cumulative_pv_count = cumulative_pv_count + np.array(df[df['year'] == year]['num_of_installations'])
    cumulative_pv_count_dict[year] = cumulative_pv_count.copy()
    
df['cum_num_of_installations'] = np.concatenate([cumulative_pv_count_dict[x] for x in cumulative_pv_count_dict])
df.sort_values(['blockgroup_FIPS', 'year'], inplace=True)

prev_year_cum = df[(df['year'] <= 2015) & (df['year'] >= 2005)][['blockgroup_FIPS', 
                                                                 'year',
                                                                 'cum_num_of_installations']]
# prev_year_cum.index = prev_year_cum['blockgroup_FIPS']
prev_year_cum.rename(columns={'cum_num_of_installations': 'cum_num_of_installations_prev'}, inplace=True)
prev_year_cum['year'] = prev_year_cum['year'] + 1

df_sub = df[(df['year'] <= 2016) & (df['year'] >= 2006)]
df_sub = pd.merge(df_sub, prev_year_cum, how='left', on=['blockgroup_FIPS', 'year'])

adoption_matrix_bg = pd.DataFrame(df_sub['num_of_installations'].to_numpy().reshape([-1, 11]))
adoption_matrix_bg.index = df_sub[df_sub['year'] == 2016]['blockgroup_FIPS']
adoption_matrix_bg.columns = [str(x) for x in range(2006, 2017)]

cum_adoption_matrix_prev_bg = pd.DataFrame(df_sub['cum_num_of_installations_prev'].to_numpy().reshape([-1, 11]))
cum_adoption_matrix_prev_bg.index = df_sub[df_sub['year'] == 2016]['blockgroup_FIPS']
cum_adoption_matrix_prev_bg.columns = [str(x) for x in range(2006, 2017)]

df_buildings = df[df['year'] == 2016][['blockgroup_FIPS', 'num_of_buildings_lt600']]
df_buildings.index = df_buildings['blockgroup_FIPS']

del df

# Normal Bass Model: Block group level curve fitting
adoption_params_dict = {}
i = 0
for bfips in tqdm(cum_adoption_matrix_prev_bg.index):
    i += 1
    y_arr = cum_adoption_matrix_prev_bg.loc[bfips, :].to_numpy()
    dy_arr = adoption_matrix_bg.loc[bfips, :].to_numpy()
    if np.any(dy_arr > 0):
        base = df_buildings.loc[bfips, 'num_of_buildings_lt600']
        base = max(base, 1.0)
        y_true = np.array(list(y_arr[1:]) + [y_arr[-1] + dy_arr[-1]])
        rmse, (p, q, m, d), y_pred, dy_pred, best_onset_idx = get_best_fit_NLS(y_arr, dy_arr,
                                                                               initial_m=min(base, y_true[-1]),
                                                                               upper_m=base) # BM
        r2 = r2_score(y_true, y_pred)
    else:
        p, q, m, d, rmse, r2 = 0, 0, y_arr[0], None, 0., 1.
    adoption_params_dict[bfips] = [p, q, m, d, rmse, r2]
    if i % 5000 == 0:
        adoption_params_bg = pd.DataFrame(adoption_params_dict).transpose()
        adoption_params_bg.columns = ['p', 'q', 'm', 'd', 'rmse', 'r2']
        adoption_params_bg.to_csv('results/bass_model/adoption_bass_model_params_bg.csv')
        
adoption_params_bg = pd.DataFrame(adoption_params_dict).transpose()
adoption_params_bg.columns = ['p', 'q', 'm', 'd', 'rmse', 'r2']
adoption_params_bg.to_csv('results/bass_model/adoption_bass_model_params_bg.csv')
