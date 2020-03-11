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
from learning import *



if __name__ == '__main__':
    # ignore warning messages
    warnings.filterwarnings("ignore")

    max_val = 9

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

    # a list of features that are selected manually to remove from the dataframe
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
    # a subset of features that were selected manually
    x_df = data_df.drop(['critical_temp'], axis=1, inplace=False)
    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df)
    # a subset of features that were selected automatically
    x_df1 = filterFeaturesByCorrelationMatrix()
    x_train1, x_test1, y_train1, y_test1 = train_test_split(x_df1, y_df)
    # a subset of features that were suggested in the given paper.
    best_20_df = getBest20Features()
    new_x_train, new_x_test, new_y_train, new_y_test = train_test_split(best_20_df, y_df)

    # a list of alpha values that are used for the regularisation
    alpha_vals = [0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 50, 100]


    # create, train, and test model

    if mode == 1:
        # mode 1 => LinearRegression

        # LinearRegression with a subset of features that were seleected manually
        print('\nCreate LinearRegression model with the custom feature set - selected manually')
        train_test_and_analyse(LinearRegression(), x_train, x_test, y_train, y_test)

        # LinearRegression with a subset of features that were seleected manually
        print('\nCreate LinearRegression model with the custom feature set - selecetd automatically')
        train_test_and_analyse(LinearRegression(), x_train1, x_test1, y_train1, y_test1)

        # LinearRegression with the features that the paper suggested
        print('\nCreate LinearRegression model with a set of suggested features')
        train_test_and_analyse(LinearRegression(), new_x_train, new_x_test, new_y_train, new_y_test)

    elif mode == 2:
        # mode 2 => Regularisation with Ridge

        print('\nRidge model with the custom feature set - selecetd manually')
        test_ridge(alpha_vals, x_train, x_test, y_train, y_test)

        print('\nRidge model with the custom feature set - selecetd automatically')
        test_ridge(alpha_vals, x_train1, x_test1, y_train1, y_test1)

        print('\nRidge model with a set of suggested features')
        test_ridge(alpha_vals, new_x_train, new_x_test, new_y_train, new_y_test)

    elif mode == 3:
        # mode 3 => Regularisation with Lasso

        print('\nLasso model with the custom feature set - selected manually')
        test_lasso(alpha_vals, x_train, x_test, y_train, y_test)

        print('\nLasso model with the custom feature set - selected automatically')
        test_lasso(alpha_vals, x_train1, x_test1, y_train1, y_test1)

        print('\nLasso model with a set of suggested features')
        test_lasso(alpha_vals, new_x_train, new_x_test, new_y_train, new_y_test)

    elif mode == 4:
        # mode 4 => ElasticNet

        print('\nCreate ElasticNet model with the custom feature set - selected manually')
        train_test_and_analyse(ElasticNet(), x_train, x_test, y_train, y_test)

        print('\nCreate ElasticNet model with the custom feature set - selected automatically')
        train_test_and_analyse(ElasticNet(), x_train1, x_test1, y_train1, y_test1)

        print('\nCreate ElasticNet model with a set of suggested features')
        train_test_and_analyse(ElasticNet(), new_x_train, new_x_test, new_y_train, new_y_test)

    elif mode == 5:
        # mode 5 => LinearRegression with PolynomialFeature

        print('\nPolynomial regression with the custom feature set - selected manually')
        make_pipeline_for_polynomial_regression(x_train, x_test, y_train, y_test)

        print('\nPolynomial regression with the custom feature set - selected automatically')
        make_pipeline_for_polynomial_regression(x_train1, x_test1, y_train1, y_test1, degrees=[1,2,3])

        print('\nPolynomial regression with a set of suggested features')
        make_pipeline_for_polynomial_regression(new_x_train, new_x_test, new_y_train, new_y_test, degrees=[1,2,3])

    elif mode == 6:
        # mode 6 => RandomForestRegressor

        print('\nCreate RandomForestRegressor model with the custom feature set - selected manually')
        train_test_and_analyse(RandomForestRegressor(), x_train, x_test, y_train, y_test)

        print('\nPolynomial regression with the custom feature set - selected automatically')
        train_test_and_analyse(RandomForestRegressor(), x_train1, x_test1, y_train1, y_test1)

        print('\nCreate RandomForestRegressor model with a set of suggested features')
        train_test_and_analyse(RandomForestRegressor(), new_x_train, new_x_test, new_y_train, new_y_test)

    elif mode == 7:
        # mode 7 => GradientBoostingRegressor

        print('\nCreate GradientBoostingRegressor model with the custom feature set - selected manually')
        train_test_and_analyse(GradientBoostingRegressor(), x_train, x_test, y_train, y_test)

        print('\nCreate GradientBoostingRegressor model with the custom feature set - selected automatically')
        train_test_and_analyse(GradientBoostingRegressor(), x_train1, x_test1, y_train1, y_test1)

        print('\nCreate GradientBoostingRegressor model with a set of suggested features')
        train_test_and_analyse(GradientBoostingRegressor(), new_x_train, new_x_test, new_y_train, new_y_test)

    elif mode == 8:
        # Print out coefficient values of LinearRegression model and ElasticNet model
        models = [LinearRegression(), ElasticNet()]

        print('\n\nGet coefficient values - Trained with a set of suggested features (selected manually)')
        for model in models:
            print('\nmodel = {}'.format(type(model).__name__))
            model.fit(x_train, y_train)
            getCoefficientValues(model, x_df)

        print('\n\nGet coefficient values - Trained with a set of suggested features (selected automatically)')
        for model in models:
            print('\nmodel = {}'.format(type(model).__name__))
            model.fit(x_train1, y_train1)
            getCoefficientValues(model, x_df1)

        print('\n\nGet coefficient values - Trained with a set of suggested features')
        for model in models:
            print('\nmodel = {}'.format(type(model).__name__))
            model.fit(new_x_train, new_y_train)
            getCoefficientValues(model, best_20_df)


    elif mode == 9:
        # K-Fold cross validation

        # a list of models to test
        models = [LinearRegression(), ElasticNet(), GradientBoostingRegressor(), RandomForestRegressor()]
        # do the cross validation with the K-Fold
        for model in models:
            train_test_and_analyse_with_kfold(model, data_df, x_df, y_df)
