import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score


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
