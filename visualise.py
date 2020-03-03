import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
from utils import load_train_csv, load_unique_m_csv


#54 > x
def generate_regplot(data_df, x_df, fig_size=(12, 30), n_cols=3, n_rows=27):
    fig, axs = plt.subplots(figsize=fig_size, ncols=n_cols, nrows=n_rows)
    for i, feature in enumerate(x_df.columns):
        row = int(i / n_cols)
        col = i % n_cols
        #sns.regplot(x=feature, y='critical_temp', data=data_df, ax=axs[row][col])
        sns.regplot(x=feature, y='critical_temp', data=data_df, ax=axs[row][col])
        plt.subplots_adjust(wspace=None, hspace=None)
    plt.show()


# Correlation matrix
def plotCorrelationMatrix(df, graphWidth, filename):
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()


# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize, annotate_plot=False):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns

    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)

    # reduce the number of columns for matrix inversion of kernel density plots
    if len(columnNames) > 10:
        columnNames = columnNames[:10]

    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values

    if annotate_plot:
        for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
            ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')
    plt.show()


if __name__ == '__main__':
    # ignore warning messages
    warnings.filterwarnings("ignore")

    max_val = 5

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

    if mode == 1:
        plotCorrelationMatrix(data_df, 20, 'train.csv')
    elif mode == 2:
        plotScatterMatrix(data_df, 30, 10)
    elif mode == 3:
        plotCorrelationMatrix(chem_df, 20, 'unique_m.csv')
    elif mode == 4:
        plotScatterMatrix(chem_df, 30, 10)
    elif mode == 5:
        x_df = data_df.drop('critical_temp', axis=1, inplace=False)
        generate_regplot(data_df, x_df)
