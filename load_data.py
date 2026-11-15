# This file contains the function to pull the CSVs into dataframes with standardized names

import pandas as pd

def load_data():
    """
    In:
        None
    Out:
        df1: df corresponding to AgNP_dataset.csv
        df2: df corresponding to AutoAM_dataset.csv
        df3: df corresponding to Crossed barrel_dataset.csv
        df4: df corresponding to P3HT_dataset.csv
        df5: df corresponding to Perovskite_dataset.csv
    Purpose:
        Load in all five datasets and standardize naming
    """

    # load in the three datasets
    df1 = pd.read_csv('data/AgNP_dataset.csv')
    df2 = pd.read_csv('data/AutoAM_dataset.csv')
    df3 = pd.read_csv('data/Crossed barrel_dataset.csv')
    df4 = pd.read_csv('data/P3HT_dataset.csv')
    df5 = pd.read_csv('data/Perovskite_dataset.csv')

    # standardize column names (o1-on for observed columns, s for real state)
    df1 = df1.rename(columns={'QAgNO3(%)': 'o1',
                              'Qpva(%)': 'o2',
                              'Qtsc(%)': 'o3',
                              'Qseed(%)': 'o4',
                              'Qtot(uL/min)': 'o5',
                              'loss': 's'})
    df2 = df2.rename(columns={'Prime Delay': 'o1',
                              'Print Speed': 'o2', 
                              'X Offset Correction': 'o3',
                              'Y Offset Correction': 'o4',
                              'Score': 's'})
    df3 = df3.rename(columns={'n': 'o1',
                              'theta': 'o2',
                              'r': 'o3',
                              't': 'o4',
                              'toughness': 's'})
    df4 = df4.rename(columns={'P3HT content (%)': 'o1',
                              'D1 content (%)': 'o2',
                              'D2 content (%)': 'o3',
                              'D6 content (%)': 'o4',
                              'D8 content (%)': 'o5',
                              'Conductivity (measured) (S/cm)': 's'})
    df5 = df5.rename(columns={'CsPbI': 'o1',
                              'FAPbI': 'o2',
                              'MAPbI': 'o3',
                              'Instability index': 's'})

    return df1, df2, df3, df4, df5