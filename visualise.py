import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
import math
from utils import load_train_csv, load_unique_m_csv



def plot_regplots(data_df, x_df, n_cols=3, n_rows=27, path='./regplot1.png'):
    """
    Plot the scatter matrix by using regplots.
    The generated plot will be saved as image file instead of displaying plots.

    :param data_df: The dataframe
    :param x_df:    The dataframe that contains all columns to visualise.
    :param n_cols:  The number of columns in subplots (default = 3)
    :param n_rows:  The number of rows in subplots (default = 27)
    :param paths:   The file path of the generated image.
    """
    fig_size = (n_cols * 5, n_rows * 4)
    fig, axs = plt.subplots(figsize=fig_size, ncols=n_cols, nrows=n_rows)
    for i, feature in enumerate(x_df.columns):
        row = int(i / n_cols)
        col = i % n_cols
        #sns.regplot(x=feature, y='critical_temp', data=data_df, ax=axs[row][col])
        sns.regplot(x=feature, y='critical_temp', data=data_df, ax=axs[row][col])
        plt.subplots_adjust(wspace=None, hspace=None)

    # Save plot to image file instead of displaying
    fig.savefig(path)


# Correlation matrix
def plotCorrelationMatrix(df, graphWidth, filename):
    """
    Plot the correlation matrix by using heatmaps.
    The generated plot will be saved as image file instead of displaying plots.

    :param df:          The dataframe
    :param graphWidth:  The width of the graph
    :param filename:    A name of the generated image file
    """

    # drop columns with NaN
    df = df.dropna('columns')
    # keep columns where there are more than 1 unique values
    df = df[[col for col in df if df[col].nunique() > 1]]

    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return

    # get the correlational matrix
    corr = df.corr()

    # plot the figure
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)

    # Save plot to image file instead of displaying
    name = filename.replace('.csv', '')
    plt.savefig('{}.png'.format(name))




if __name__ == '__main__':
    # ignore warning messages
    warnings.filterwarnings("ignore")

    max_val = 4

    if len(sys.argv) < 2:
        print('Usage: python3 visualise.py <mode_number>')
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


    if mode == 1:
        plotCorrelationMatrix(data_df, 20, 'train.csv')
    elif mode == 2:
        x_df = data_df.drop('critical_temp', axis=1, inplace=False)
        plot_regplots(data_df, x_df)
    elif mode == 3:
        plotCorrelationMatrix(chem_df, 20, 'unique_m.csv')
    elif mode == 4:
        x_df = chem_df.drop('critical_temp', axis=1, inplace=False)
        x_df = x_df.select_dtypes(include=[np.number])
        num_of_rows = math.ceil(x_df.shape[1] / 3)
        plot_regplots(chem_df, x_df, n_cols=3, n_rows=num_of_rows, path='./regplot2.png')
