#!/usr/bin/env python

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from utils import load_train_csv, load_unique_m_csv, getBest20Features
from learning import train_test_and_analyse, test_ridge, test_lasso



if __name__ == '__main__':
    # ignore warning messages
    warnings.filterwarnings("ignore")

    # Load and clean data

    data_df = load_train_csv()
    chem_df = load_unique_m_csv()

    #TODO
    less_correlated_list = ["gmean_fie", "mean_atomic_radius", "mean_ElectronAffinity",
                            "range_FusionHeat", "std_FusionHeat", "mean_FusionHeat",
                            "wtd_std_FusionHeat", "range_Valence", "std_Valence",
                            "wtd_std_Valence", "wtd_mean_FusionHeat", "gmean_FusionHeat",
                            "wtd_gmean_FusionHeat", "wtd_range_FusionHeat", "wtd_mean_atomic_radius",
                            "gmean_atomic_radius", "wtd_gmean_atomic_radius", "mean_atomic_mass",
                            "gmean_atomic_mass", "wtd_gmean_atomic_mass", "wtd_range_atomic_mass", 
                            "wtd_mean_fie", "wtd_gmean_fie", "wtd_std_fie", "wtd_mean_Density", 
                            "wtd_range_Density", "gmean_ThermalConductivity", "wtd_range_Valence"]

    data_df.drop(less_correlated_list, axis=1, inplace=True)
    print('Drop less correlated features from the dataframe')

    # get train set and test set
    y_df = data_df['critical_temp']
    x_df = data_df.drop(['critical_temp'], axis=1, inplace=False)
    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df)

    # LinearRegression
    print('\nCreate LinearRegression model')
    lr1 = train_test_and_analyse(LinearRegression(), x_train, x_test, y_train, y_test)

    #TODO
    print('\nCreate LinearRegression model with best 20 features')
    best_20_df = getBest20Features()
    new_x_train, new_x_test, new_y_train, new_y_test = train_test_split(best_20_df, y_df)
    lr2 = train_test_and_analyse(LinearRegression(), new_x_train, new_x_test, new_y_train, new_y_test)

    alpha_vals = [0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 50, 100]
    test_ridge(alpha_vals, x_train, x_test, y_train, y_test)
    test_ridge(alpha_vals, new_x_train, new_x_test, new_y_train, new_y_test) #TODO

    test_lasso(alpha_vals, x_train, x_test, y_train, y_test)
    test_lasso(alpha_vals, new_x_train, new_x_test, new_y_train, new_y_test) #TODO

    # ElasticNet
    print('\nCreate ElasticNet model')
    el_net = train_test_and_analyse(ElasticNet(), x_train, x_test, y_train, y_test)

    # RandomForestRegressor
    print('\nCreate RandomForestRegressor model')
    rfr = train_test_and_analyse(RandomForestRegressor(), x_train, x_test, y_train, y_test)

    # GradientBoostingRegressor
    print('\nCreate GradientBoostingRegressor model')
    gbr = train_test_and_analyse(GradientBoostingRegressor(), x_train, x_test, y_train, y_test)
