import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import random

file = Path.cwd().joinpath('HW2_data.csv')
T1D_dataset = pd.read_csv(file)
# Preprocessing
random.seed(10)
from HW2_functions import rm_nan_pat as rm
T1D_df_str = rm(T1D_dataset)
col_names = T1D_df_str.columns
T1D_df = pd.get_dummies(T1D_df_str, drop_first=True)
cols = T1D_df.columns.tolist()
cols = cols[0:1] + cols[2:17] + cols[1:2] + cols[-1:]
T1D_df = T1D_df[cols]
T1D_df.columns = ['Age', 'Gender', 'Increased Urination', 'Increased Thirst',
       'Sudden Weight Loss', 'Weakness', 'Increased Hunger', 'Genital Thrush',
       'Visual Blurring', 'Itching', 'Irritability', 'Delayed Healing',
       'Partial Paresis', 'Muscle Stiffness', 'Hair Loss', 'Obesity',
       'Family History', 'Diagnosis']


T1D_feats = T1D_df.copy()
del T1D_feats['Diagnosis']
Diagnosis = T1D_df[['Diagnosis']]
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(T1D_feats,np.ravel(Diagnosis), test_size=0.2, random_state=0, stratify=np.ravel(Diagnosis))

from HW2_functions import col_charts as c_ch
feat_remove = ['Age']
c_ch(X_train,X_test,feat_remove, y_train,y_test)

from HW2_functions import feat_lable as ft_lb
remove_feat = 'Age'
ft_lb(X_train,remove_feat, y_train)

# scaling 'Age' column
from HW2_functions import cv_kfold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
scaler = MinMaxScaler()
# #scaler = StandardScaler()
# features_df = X_train.copy()
# col_to_scale = ['Age']
# feat_to_scale = features_df[col_to_scale]
# scaler = MinMaxScaler().fit(feat_to_scale.values)
# scaled_feats = scaler.transform(feat_to_scale.values)
# X_train[col_to_scale] = scaled_feats
X_train[['Age']] = scaler.fit_transform(X_train[['Age']])
X_test[['Age']] = scaler.transform(X_test[['Age']])

C = [0.001,0.01,0.25,0.5,0.75,1,1.25,1.5,1.75,2,3,4,5,10,20,50] # make a list of up to 6 different values of regularization parameters and examine their effects
K = 5 # choose a number of folds
val_dict = cv_kfold(X_train, y_train, C=C, penalty=['l1', 'l2'], K=K)
for idx, elem in enumerate(val_dict):
       print("For C=%.3f and penalty=%s the AUC is: %.3f" % (val_dict[idx].get('C'), val_dict[idx].get('penalty'), val_dict[idx].get('AUC')))

from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss

C = 1 
penalty = 'l2' 
logreg = LogisticRegression(solver='saga', multi_class='ovr', penalty=penalty, C=C, max_iter=10000)
logreg.fit(X_train, y_train)

y_pred_train = logreg.predict(X_train)
y_pred_proba_train = logreg.predict_proba(X_train)
AUC_train = roc_auc_score(y_train, y_pred_proba_train[:,1])
loss_train = log_loss(y_train, y_pred_train)

y_pred_test = logreg.predict(X_test)
y_pred_proba_test = logreg.predict_proba(X_test)
AUC_test = roc_auc_score(y_test, y_pred_proba_test[:,1])
loss_test = log_loss(y_test, y_pred_test)

cnf_matrix_train = metrics.confusion_matrix(y_pred_train, y_train)
ax1 = plt.subplot()
sns.heatmap(cnf_matrix_train, annot=True, xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
ax1.set(ylabel='True labels', xlabel='Predicted labels')
plt.show()

print('Evaluation metrics for the training set:')
print('AUC is: %.3f' % (AUC_train))
print('The loss is: %.3f' % (loss_train))
print("F1 score is: " + str("{0:.2f}".format(100 * metrics.f1_score(y_train, y_pred_train, average='macro'))) + "%")
print("Accuracy is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(y_train, y_pred_train))) + "%")

cnf_matrix_test = metrics.confusion_matrix(y_pred_test, y_test)
ax1 = plt.subplot()
sns.heatmap(cnf_matrix_test, annot=True, xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
ax1.set(ylabel='True labels', xlabel='Predicted labels')
plt.show()

print('Evaluation metrics for the test set:')
print('AUC is: %.3f' % (AUC_test))
print('The loss is: %.3f' % (loss_test))
print("F1 score is: " + str("{0:.2f}".format(100 * metrics.f1_score(y_test, y_pred_test, average='macro'))) + "%")
print("Accuracy is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(y_test, y_pred_test))) + "%")

from sklearn.model_selection import StratifiedKFold
n_splits = 3
skf = StratifiedKFold(n_splits=n_splits, random_state=10, shuffle=True)

from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import GridSearchCV
# svc = SVC(probability=True)
svc = svm.SVC(probability=True)
C = np.array([0.001, 0.01, 1, 10, 100, 1000])
svm_lin = GridSearchCV(estimator=svc,
             param_grid={'svm__kernel':['linear'], 'svm__C':C},
             scoring=['accuracy','f1','precision','recall','roc_auc'],
             cv=skf, refit='roc_auc', verbose=3, return_train_score=True)
svm_lin.fit(X_train, y_train)
best_svm_lin = svm_lin.best_estimator_
print(svm_lin.best_params_)
# clf_type = ['linear']
# plot_radar(svm_lin,clf_type)
print('hi')
print('hi')
