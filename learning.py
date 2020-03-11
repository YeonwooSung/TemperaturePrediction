import numpy as np
import pandas as pd
import time
import warnings
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from utils import load_train_csv, getBest20Features, filterFeaturesByCorrelationMatrix



def train_test_and_analyse(model, x_train, x_test, y_train, y_test):
    """
    Train the given model with given dataset.
    Then, evalutate the given model by calculating MSE, RMSE, and R^2 score.
    """
    model.fit(x_train, y_train)
    y_preds = model.predict(x_test)
    mse = mean_squared_error(y_test, y_preds)
    rmse = np.sqrt(mse)
    variance_score = r2_score(y_test, y_preds)
    print('MSE = {0:.3f}\nRMSE = {1:.3f}\nR2 score = {2:.3f}'.format(mse, rmse, variance_score))
    
    return model


def train_test_and_analyse_with_kfold(model, df, x_df, y_df, num_of_splits=5):
    """
    Run the K-Fold validation testing for the given model.
    """
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
    """
    Train the ridge model with the given dataset.
    Then, call the train_test_and_analyse() function for the evaluation.
    """
    print('\nCreate and test Ridge models')
    for alpha_val in alpha_vals:
        print('Ridge(alpha={})'.format(alpha_val))
        ridge = Ridge(alpha=alpha_val)
        train_test_and_analyse(ridge, x_train, x_test, y_train, y_test)
        print()


def test_lasso(alpha_vals, x_train, x_test, y_train, y_test):
    """
    Train the lasso model with the given dataset.
    Then, call the train_test_and_analyse() function for the evaluation.
    """
    print('\nCreate and test Lasso models')
    for alpha_val in alpha_vals:
        print('Lasso(alpha={})'.format(alpha_val))
        lasso = Lasso(alpha=alpha_val)
        train_test_and_analyse(lasso, x_train, x_test, y_train, y_test)
        print()


def make_pipeline_for_polynomial_regression(x_train, x_test, y_train, y_test, degrees=[1, 2]):
    """
    Makes a pipeline with PolynomialFeatures and LinearRegression, so that the user could perform
    the polynomial regression.
    """
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


def visualise_3_linear_models(x_df1, x_df2, x_df3, y_df, fig_name1='selected manually', fig_name2='selected automatically', fig_name3='suggested features', file_name='./linear_regressions.png'):
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(16, 4), ncols=3)

    ax1.set_title(fig_name1)
    lr1 = LinearRegression()
    lr1.fit(x_df1, y_df)
    ax1.scatter(y_df, lr1.predict(x_df1), c='b')
    ax1.plot(y_df, y_df, c='m')

    ax2.set_title(fig_name2)
    lr2 = LinearRegression()
    lr2.fit(x_df2, y_df)
    ax2.scatter(y_df, lr2.predict(x_df2), c='b')
    ax2.plot(y_df, y_df, c='m')

    ax3.set_title(fig_name3)
    lr3 = LinearRegression()
    lr3.fit(x_df3, y_df)
    ax3.scatter(y_df, lr3.predict(x_df3), c='b')
    ax3.plot(y_df, y_df, c='m')

    # Save plot to image file instead of displaying
    fig.savefig(file_name)



if __name__ == '__main__':
    # ignore warning messages
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    if len(sys.argv) < 2:
        print('Usage: python3 utils.py <mode_number>')
        exit(1)
    try:
        mode = int(sys.argv[1])
        if mode > 3:
            raise ValueError
    except ValueError:
        print('The first argument should be one of 1, 2, and 3')
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

    # extract answer column from a dataframe
    y_df = data_df['critical_temp']


    # get dataframes that are filtered with correlation values
    filtered_df1 = filterFeaturesByCorrelationMatrix()


    if mode == 1:
        # Visualise 3 linear regression model where the first model is trained with the
        # subset of features that are selected manually, the second model is trained with the
        # suggested features, and the third model is trined with the subset of features that are
        # selected automatically.

        # get train set and test set
        x_df = data_df.drop(['critical_temp'], axis=1, inplace=False)
        x_train, x_test, y_train, y_test = train_test_split(x_df, y_df)
        best_20_df = getBest20Features()

        # visualise 3 linear regression models, where each model is trained with different set of features
        visualise_3_linear_models(x_df, filtered_df1, best_20_df, y_df)


    elif mode == 2:
        # Cross validation for 3 subsets of automatically selected features, where
        # all 3 of them uses different threshold value for choosing features.

        filtered_df2 = filterFeaturesByCorrelationMatrix(cor_thres=0.15)
        filtered_df3 = filterFeaturesByCorrelationMatrix(cor_thres=0.2)

        # visualise 3 linear regression models, where each model is trained with different set of features
        visualise_3_linear_models(filtered_df1, filtered_df2, filtered_df3, y_df, fig_name1='correlation_thres=0.1', fig_name2='correlation_thres=0.15', fig_name3='correlation_thres=0.2', file_name='./cor_linear_regression.png')


    else:
        # Cross validation for 3 subsets of automatically selected features, where
        # all 3 of them uses different threshold value for choosing features.

        filtered_df2 = filterFeaturesByCorrelationMatrix(thres=0.55)
        filtered_df3 = filterFeaturesByCorrelationMatrix(thres=0.6)

        # visualise 3 linear regression models, where each model is trained with different set of features
        visualise_3_linear_models(filtered_df1, filtered_df2, filtered_df3, y_df, fig_name1='threshld=0.5', fig_name2='threshld=0.55', fig_name3='threshld=0.6', file_name='./thres_linear_regression.png')
