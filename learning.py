import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso
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

def cross_validate_gridsearch(model, params, x_train, x_test, y_train, y_test):
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
    print('===============================================================')


if __name__ == '__main__':
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
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    learning_rate = [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    subsamples = [0.9, 0.5, 0.3, 0.2]


    # use GridSearchCV to find the best hyperparameters

    params_rfr = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap
    }
    cross_validate_gridsearch(RandomForestRegressor(), params_rfr, x_train, x_test, y_train, y_test)

    params_gbr = {
        'learning_rate': learning_rate,
        'subsample': subsamples,
        'n_estimators': n_estimators,
        'max_depth': max_depth
    }
    cross_validate_gridsearch(GradientBoostingRegressor(), params_gbr, x_train, x_test, y_train, y_test)
