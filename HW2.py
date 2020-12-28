import numpy as np
import pandas as pd
from pathlib import Path
import random

file = Path.cwd().joinpath('HW2_data.csv')
T1D_dataset = pd.read_csv(file)
# Preprocessing
from HW2_functions import rm_nan_pat as rm
T1D_pr = rm(T1D_dataset)

print('hi')
print('hi')