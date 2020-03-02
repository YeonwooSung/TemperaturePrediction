import numpy as np
import pandas as pd
import os
import sys



def load_dataframe(file_path, debug_mode=False):
    if os.path.exists(file_path) and os.path.isfile(file_path):
        print('load dataframe from "{}"'.format(file_path))
    else:
        print('Invalid file path!')
        return None
    data_df = pd.read_csv(file_path)

    if debug_mode:
        print('dataframe.head(10) : ', data_df.head(10))
        print('\ndataframe.shape : {}'.format(data_df.shape))
        print('\ndataframe.info() : ')
        data_df.info()
        print('\ndataframe.describe() : ', data_df.describe())
        print()

    return data_df


def clean_data(data_df, intFeatures=[], floatFeatures=[], stringFeatures=[]):
    # Assign default values for each data type
    defaultInt = 0
    defaultString = 'NaN'
    defaultFloat = 0.0

    # Clean the NaN's
    for feature in data_df:
        if feature in intFeatures:
            data_df[feature] = data_df[feature].fillna(defaultInt)
        elif feature in stringFeatures:
            data_df[feature] = data_df[feature].fillna(defaultString)
        elif feature in floatFeatures:
            data_df[feature] = data_df[feature].fillna(defaultFloat)
        else:
            print('Error: Feature %s not recognized.' % feature)

    return data_df


def clean_NaN_for_unique_m(chem_df):
    for feature in chem_df:
        if feature != 'material':
            chem_df[feature] = chem_df[feature].fillna(0)
    chem_df.fillna(chem_df.mean(), inplace=True)
    print('clean NaN values in unique_m')
    return chem_df


def load_train_csv(debug_mode=False):
    # Load data
    data_df = load_dataframe('./data/train.csv', debug_mode)

    # Clean data
    # create lists by data type
    intFeatures = ['number_of_elements',
                   'range_atomic_radius', 'range_Valence']

    floatFeatures = ['mean_atomic_mass', 'wtd_mean_atomic_mass',
                     'gmean_atomic_mass', 'wtd_gmean_atomic_mass', 'entropy_atomic_mass',
                     'wtd_entropy_atomic_mass', 'range_atomic_mass', 'wtd_range_atomic_mass',
                     'std_atomic_mass', 'wtd_std_atomic_mass', 'mean_fie', 'wtd_mean_fie',
                     'gmean_fie', 'wtd_gmean_fie', 'entropy_fie', 'wtd_entropy_fie',
                     'range_fie', 'wtd_range_fie', 'std_fie', 'wtd_std_fie',
                     'mean_atomic_radius', 'wtd_mean_atomic_radius', 'gmean_atomic_radius',
                     'wtd_gmean_atomic_radius', 'entropy_atomic_radius',
                     'wtd_entropy_atomic_radius',
                     'wtd_range_atomic_radius', 'std_atomic_radius', 'wtd_std_atomic_radius',
                     'mean_Density', 'wtd_mean_Density', 'gmean_Density',
                     'wtd_gmean_Density', 'entropy_Density', 'wtd_entropy_Density',
                     'range_Density', 'wtd_range_Density', 'std_Density', 'wtd_std_Density',
                     'mean_ElectronAffinity', 'wtd_mean_ElectronAffinity',
                     'gmean_ElectronAffinity', 'wtd_gmean_ElectronAffinity',
                     'entropy_ElectronAffinity', 'wtd_entropy_ElectronAffinity',
                     'range_ElectronAffinity', 'wtd_range_ElectronAffinity',
                     'std_ElectronAffinity', 'wtd_std_ElectronAffinity', 'mean_FusionHeat',
                     'wtd_mean_FusionHeat', 'gmean_FusionHeat', 'wtd_gmean_FusionHeat',
                     'entropy_FusionHeat', 'wtd_entropy_FusionHeat', 'range_FusionHeat',
                     'wtd_range_FusionHeat', 'std_FusionHeat', 'wtd_std_FusionHeat',
                     'mean_ThermalConductivity', 'wtd_mean_ThermalConductivity',
                     'gmean_ThermalConductivity', 'wtd_gmean_ThermalConductivity',
                     'entropy_ThermalConductivity', 'wtd_entropy_ThermalConductivity',
                     'range_ThermalConductivity', 'wtd_range_ThermalConductivity',
                     'std_ThermalConductivity', 'wtd_std_ThermalConductivity',
                     'mean_Valence', 'wtd_mean_Valence', 'gmean_Valence',
                     'wtd_gmean_Valence', 'entropy_Valence', 'wtd_entropy_Valence',
                     'wtd_range_Valence', 'std_Valence', 'wtd_std_Valence',
                     'critical_temp']

    data_df = clean_data(data_df, intFeatures, floatFeatures, [])

    return data_df


def load_unique_m_csv(debug_mode=False):
    # Load data
    chem_df = load_dataframe('./data/unique_m.csv', debug_mode)

    # Clean data
    for feature in chem_df:
        if feature != 'material':
            chem_df[feature] = chem_df[feature].fillna(0)
    chem_df.fillna(chem_df.mean(), inplace=True)

    return chem_df


def getBest20Features():
    most_important_list_20 = ["range_ThermalConductivity", "wtd_std_ThermalConductivity", "range_atomic_radius",
                              "wtd_gmean_ThermalConductivity", "std_ThermalConductivity", "wtd_entropy_Valence",
                              "wtd_std_ElectronAffinity", "wtd_entropy_atomic_mass", "wtd_mean_Valence",
                              "wtd_gmean_ElectronAffinity", "wtd_range_ElectronAffinity", "wtd_mean_ThermalConductivity",
                              "wtd_gmean_Valence", "std_atomic_mass", "std_Density",
                              "wtd_entropy_ThermalConductivity", "wtd_range_ThermalConductivity", "wtd_mean_atomic_mass",
                              "wtd_std_atomic_mass", "gmean_Density"
                              ]


    newData_df = pd.read_csv('./data/train.csv')
    most_important_20_df = newData_df[most_important_list_20]

    return most_important_20_df


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 main.py <mode_number>')
        exit(1)
    try:
        mode = int(sys.argv[1])
        if mode > 3:
            raise ValueError
    except ValueError:
        print('The first argument should be either 1 or 2')
        exit(1)

    if mode == 1:
        load_train_csv(True)
    else:
        load_unique_m_csv(True)
