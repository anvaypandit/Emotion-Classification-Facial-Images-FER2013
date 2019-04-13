import numpy as np
import pandas as pd
import os

# Loads a pandas dataframe of train data test data and validation data
def load_train_data():
    data_dir = 'data'
    train_fp = os.path.join(os.getcwd(),data_dir)
    train_fp = os.path.join(train_fp,'Train_Data.csv')
    train_df = pd.read_csv(train_fp)
    return train_df


def load_test_data():
    data_dir = 'data'
    test_fp = os.path.join(os.getcwd(),data_dir)
    test_fp = os.path.join(test_fp,'Test_Data.csv')
    test_df = pd.read_csv(test_fp)
    return test_df


def load_validation_data():
    data_dir = 'data'
    val_fp = os.path.join(os.getcwd(),data_dir)
    val_fp = os.path.join(val_fp,'Validation_Data.csv')
    val_df = pd.read_csv(val_fp)
    return val_df


def load_all_data():
    train_df = load_train_data()
    test_df = load_test_data()
    val_df = load_validation_data()
    return train_df,test_df,val_df

