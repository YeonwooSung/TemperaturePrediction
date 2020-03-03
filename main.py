#!/usr/bin/env python

import pandas as pd
import numpy as np
import warnings
import math
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from utils import load_train_csv, load_unique_m_csv, getBest20Features
from learning import train_test_and_analyse, test_ridge, test_lasso, make_pipeline_for_polynomial_regression



if __name__ == '__main__':
    # ignore warning messages
    warnings.filterwarnings("ignore")

    max_val = 7

    if len(sys.argv) < 2:
        print('Usage: python3 main.py <mode_number>')
        exit(1)
    try:
        mode = int(sys.argv[1])
        if mode > max_val:
            raise ValueError
    except ValueError:
        print('The first argument should be one of 1 ~ {}'.format(max_val))
        exit(1)

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
    best_20_df = getBest20Features()
    new_x_train, new_x_test, new_y_train, new_y_test = train_test_split(best_20_df, y_df)

    # a list of alpha values that are used for the regularisation
    alpha_vals = [0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 50, 100]

    if mode == 1:
        # mode 1 => LinearRegression

        # LinearRegression with custom features
        print('\nCreate LinearRegression model with the custom feature set')
        train_test_and_analyse(LinearRegression(), x_train, x_test, y_train, y_test)

        # LinearRegression with the features that the paper suggested
        print('\nCreate LinearRegression model with a set of suggested features')
        train_test_and_analyse(LinearRegression(), new_x_train, new_x_test, new_y_train, new_y_test)

    elif mode == 2:
        # mode 2 => Regularisation with Ridge

        print('\nRidge model with the custom feature set')
        test_ridge(alpha_vals, x_train, x_test, y_train, y_test)

        print('\nRidge model with a set of suggested features')
        test_ridge(alpha_vals, new_x_train, new_x_test, new_y_train, new_y_test)

    elif mode == 3:
        # mode 3 => Regularisation with Lasso

        print('\nLasso model with the custom feature set')
        test_lasso(alpha_vals, x_train, x_test, y_train, y_test)

        print('\nLasso model with a set of suggested features')
        test_lasso(alpha_vals, new_x_train, new_x_test, new_y_train, new_y_test)

    elif mode == 4:
        # mode 4 => ElasticNet
        print('\nCreate ElasticNet model with the custom feature set')
        train_test_and_analyse(ElasticNet(), x_train, x_test, y_train, y_test)

        print('\nCreate ElasticNet model with a set of suggested features')
        train_test_and_analyse(ElasticNet(), new_x_train, new_x_test, new_y_train, new_y_test)
    elif mode == 5:
        # mode 5 => LinearRegression with PolynomialFeature

        print('\nPolynomial regression with the custom feature set')
        make_pipeline_for_polynomial_regression(x_train, x_test, y_train, y_test)

        print('\nPolynomial regression with a set of suggested features')
        make_pipeline_for_polynomial_regression(new_x_train, new_x_test, new_y_train, new_y_test)

    elif mode == 6:
        # mode 6 => RandomForestRegressor

        print('\nCreate RandomForestRegressor model with the custom feature set')
        train_test_and_analyse(RandomForestRegressor(), x_train, x_test, y_train, y_test)

        print('\nCreate RandomForestRegressor model with a set of suggested features')
        train_test_and_analyse(ElasticNet(), new_x_train, new_x_test, new_y_train, new_y_test)
    elif mode == 7:
        # mode 7 => GradientBoostingRegressor

        print('\nCreate GradientBoostingRegressor model with the custom feature set')
        train_test_and_analyse(GradientBoostingRegressor(), x_train, x_test, y_train, y_test)

        print('\nCreate GradientBoostingRegressor model with a set of suggested features')
        train_test_and_analyse(GradientBoostingRegressor(), new_x_train, new_x_test, new_y_train, new_y_test)
