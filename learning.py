import numpy as np
import pandas as pd
import time
import warnings
import sys
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, KFold
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from utils import load_train_csv



def train_test_and_analyse(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    y_preds = model.predict(x_test)
    mse = mean_squared_error(y_test, y_preds)
    rmse = np.sqrt(mse)
    variance_score = r2_score(y_test, y_preds)
    print('MSE = {0:.3f}\nRMSE = {1:.3f}\nVariance score = {2:.3f}'.format(mse, rmse, variance_score))
    
    return model


def train_test_and_analyse_with_kfold(model, df, x_df, y_df, num_of_splits=5):
    kfold = KFold(n_splits=num_of_splits)
    n_iter = 1
    cv_accuracy = []

    print('\nKFold validation with {}'.format(type(model).__name__))

    # use for loop to split and process the k-fold validation
    for train, test in kfold.split(df):
        x_train, x_test = x_df.iloc[train], x_df.iloc[test]
        y_train, y_test = y_df.iloc[train], y_df.iloc[test]

        # train and predict
        print('num_of_iteration={}'.format(n_iter))
        print('train_size={}, test_size={}'.format(x_train.shape[0], x_test.shape[0]))

        model.fit(x_train, y_train)
        y_preds = model.predict(x_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_preds))

        print('RMSE = {:.3f}'.format(rmse))
        cv_accuracy.append(rmse)
        n_iter += 1

    # calculate the mean of the rmse values
    print('Average validation accuracy = {:.3f}'.format(np.mean(cv_accuracy)))


def test_ridge(alpha_vals, x_train, x_test, y_train, y_test):
    print('\nCreate and test Ridge models')
    for alpha_val in alpha_vals:
        print('Ridge(alpha={})'.format(alpha_val))
        ridge = Ridge(alpha=alpha_val)
        train_test_and_analyse(ridge, x_train, x_test, y_train, y_test)
        print()


def test_lasso(alpha_vals, x_train, x_test, y_train, y_test):
    print('\nCreate and test Ridge models')
    for alpha_val in alpha_vals:
        print('Lasso(alpha={})'.format(alpha_val))
        lasso = Lasso(alpha=alpha_val)
        train_test_and_analyse(lasso, x_train, x_test, y_train, y_test)
        print()


def make_pipeline_for_polynomial_regression(x_train, x_test, y_train, y_test, degrees=[1, 2]):
    for degree in degrees:
        print('Linear Regression with PolynomialFeatures - degree={}'.format(degree))
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        train_test_and_analyse(model, x_train, x_test, y_train, y_test)


def getCoefficientValues(model, x_df):
    # get the coefficient value of each feature
    coeff = pd.Series(data=np.round(model.coef_, 3), index=x_df.columns)
    # sort coefficient values by descending order
    coeff = coeff.sort_values(ascending=False)
    # print out the dataframe
    print(coeff)


def cross_validate_gridsearch(model, params, x_train, x_test, y_train, y_test):
    start_t = time.time()
    print('===============================================================')
    print('GridSearchCV for {}'.format(type(model).__name__))
    total_n_of_combs = 1
    for key in params:
        total_n_of_combs = total_n_of_combs * len(params[key])
    print('The number of all combinations of hyperparameters: {}'.format(total_n_of_combs))
    g_cv = GridSearchCV(model, params, cv=5)
    g_cv.fit(x_train, y_train)

    print('===============================================================')
    print('Result from GridSearchCV')
    print('===============================================================')
    print('The best estimator: ', g_cv.best_estimator_)
    print('\nThe best parameters across ALL searched params: ', g_cv.best_params_)
    print('\nThe best score across ALL searched params: {}'.format(g_cv.best_score_))
    print('\nTotal cost time = {0:.2f}'.format(time.time() - start_t))
    print('===============================================================')


if __name__ == '__main__':
    # ignore warning messages
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    max_val = 3

    if len(sys.argv) < 2:
        print('Usage: python3 learning.py <mode_number>')
        exit(1)
    try:
        mode = int(sys.argv[1])
        if mode > max_val:
            raise ValueError
    except ValueError:
        print('The first argument should be one of 1 ~ {}'.format(max_val))
        exit(1)


    # load and clean the data
    data_df = load_train_csv()
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

    # get train set and test set
    y_df = data_df['critical_temp']
    x_df = data_df.drop(['critical_temp'], axis=1, inplace=False)
    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df)


    if mode == 1:
        print()
    
    elif mode == 2:
        # Validate the ensemble model with GridSearchCV

        # lists of hyper parameters
        n_estimators = [100, 200, 500, 1000]
        max_depth = [10, 50, 100]
        bootstrap = [True, False]

        # use GridSearchCV to find the best hyperparameters

        params_rfr = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'bootstrap': bootstrap
        }
        cross_validate_gridsearch(RandomForestRegressor(), params_rfr, x_train, x_test, y_train, y_test)

        #TODO
        learning_rate = [0.1, 0.2, 0.3]
        params_gbr = {
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'max_depth': max_depth
        }
        cross_validate_gridsearch(GradientBoostingRegressor(), params_gbr, x_train, x_test, y_train, y_test)

    elif mode == 3:
        # Validate the ensemble model with GridSearchCV

        # lists of hyper parameters
        n_estimators = [100, 200, 500, 1000]
        max_depth = [10, 50, 100]
        learning_rate = [0.1, 0.2, 0.3]

        params_gbr = {
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'max_depth': max_depth
        }
        cross_validate_gridsearch(GradientBoostingRegressor(), params_gbr, x_train, x_test, y_train, y_test)
