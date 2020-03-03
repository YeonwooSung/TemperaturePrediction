import numpy as np
import time
import warnings
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
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

    # lists of hyper parameters
    n_estimators = [100, 200, 500, 1000]
    max_depth = [10, 20, 50, 100]
    bootstrap = [True, False]
    learning_rate = [0.05, 0.1, 0.2, 0.3]


    # use GridSearchCV to find the best hyperparameters

    params_rfr = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'bootstrap': bootstrap
    }
    cross_validate_gridsearch(RandomForestRegressor(), params_rfr, x_train, x_test, y_train, y_test)

    params_gbr = {
        'learning_rate': learning_rate,
        'n_estimators': n_estimators,
        'max_depth': max_depth
    }
    cross_validate_gridsearch(GradientBoostingRegressor(), params_gbr, x_train, x_test, y_train, y_test)
