import pandas as pd


def get_training_data(file_path='./data/train', usecolumns=[]):
    training_data = pd.read_csv(file_path, skiprows=1, header=None, index_col=None)

    #TODO split dataset into training set and testing set, and x and y

    return training_data
