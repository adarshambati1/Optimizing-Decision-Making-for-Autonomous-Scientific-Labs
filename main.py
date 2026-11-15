import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from load_data import load_data, split_data

def main():
    """
    In:
        None
    Out:
        None
    Purpose:
        Coordinate running everything
    """

    # load in each dataset
    df1, df2, df3, df4, df5 = load_data()
    datasets = [df1, df2, df3, df4, df5]

    # split into train and test sets
    train_sets = []
    test_sets = []
    for dataset in datasets:
        train, test = split_data(dataset)
        train_sets.append(train)
        test_sets.append(test)

    return 0

if __name__ == "__main__":
    main()