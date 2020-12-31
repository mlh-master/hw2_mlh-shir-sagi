import numpy as np
import pandas as pd
from pathlib import Path
import random

file = Path.cwd().joinpath('HW2_data.csv')
T1D_dataset = pd.read_csv(file)
# Preprocessing
random.seed(10)
from HW2_functions import rm_nan_pat as rm
T1D_df = rm(T1D_dataset)
T1D_feats = T1D_df.copy()
del T1D_feats['Diagnosis']
Diagnosis = T1D_df[['Diagnosis']]
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(T1D_feats,np.ravel(Diagnosis), test_size=0.2, random_state=0, stratify=np.ravel(Diagnosis))

from HW2_functions import col_charts as c_ch
feat_remove = ['Age', 'Gender']
c_ch(X_train,X_test,feat_remove, y_train,y_test)

from HW2_functions import feat_lable as ft_lb
remove_feat = 'Age'
ft_lb(X_train,remove_feat, y_train)



print('hi')
print('hi')
