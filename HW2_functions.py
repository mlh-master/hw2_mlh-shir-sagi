import numpy as np
import pandas as pd
from pathlib import Path
import random

def rm_nan_pat(dataframe):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A dictionary of clean CTG called c_ctg
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    df = dataframe.copy()
    num_of_nan = df.isnull().sum(axis=1).tolist()
    rm_rows = []
    for idx ,nan_num in enumerate(num_of_nan):
        if nan_num>1:
            df.drop(df.index[idx])
            # rm_rows.append(idx)
    num_of_nan2 = df.isnull().sum(axis=0)

    print('hi')
    # df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    # c_ctg = df.to_dict('list')
    # del c_ctg[extra_feature]
    # c_ctg = {k: [elem for elem in v if pd.notnull(elem)] for k, v in c_ctg.items()}
    # --------------------------------------------------------------------------
    return df