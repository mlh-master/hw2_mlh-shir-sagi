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

# # T1D_df['Age'].value_counts().plot.bar()
# # plt.show()
# blabla = T1D_df.loc[T1D_df['Diagnosis'] == 1]
Age_pos = sns.barplot(x=T1D_df.loc[T1D_df['Diagnosis'] == 1].Age.value_counts().index, y=T1D_df.loc[T1D_df['Diagnosis'] == 1].Age.value_counts(),palette="Blues_d")
Age_pos.set_xticklabels(Age_pos.get_xticklabels(),
                          rotation=90,)
Age_pos.set_title('Age distribution of positive diagnosis population')
Age_pos.set_ylabel('Count')
Age_pos.set_xlabel('Age')
plt.show()

Age_neg = sns.barplot(x=T1D_df.loc[T1D_df['Diagnosis'] == 0].Age.value_counts().index, y=T1D_df.loc[T1D_df['Diagnosis'] == 0].Age.value_counts(),palette="Blues_d")
Age_neg.set_xticklabels(Age_neg.get_xticklabels(),
                          rotation=90)
Age_pos.set_title('Age distribution of negative diagnosis population')
Age_pos.set_ylabel('Count')
Age_pos.set_xlabel('Age')
plt.show()



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
X_train_notscl = X_train.copy()
X_test_notscl = X_test.copy()
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

C = [0.001,0.01,0.5,1,1.5,2,3,4,5,10,20,50,100,1000]# make a list of up to 6 different values of regularization parameters and examine their effects
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

y_pred_train_LR = logreg.predict(X_train)
y_pred_proba_train_LR = logreg.predict_proba(X_train)
AUC_train_LR = roc_auc_score(y_train, y_pred_proba_train_LR[:,1])
loss_train_LR = log_loss(y_train, y_pred_train_LR)

y_pred_test_LR = logreg.predict(X_test)
y_pred_proba_test_LR = logreg.predict_proba(X_test)
AUC_test_LR = roc_auc_score(y_test, y_pred_proba_test_LR[:,1])
loss_test_LR = log_loss(y_test, y_pred_test_LR)

cnf_matrix_train = metrics.confusion_matrix(y_pred_train_LR, y_train)
ax1 = plt.subplot()
sns.heatmap(cnf_matrix_train, annot=True, xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
ax1.set(ylabel='True labels', xlabel='Predicted labels')
plt.show()

print('Evaluation metrics for the training set:')
print('AUC is: %.3f' % (AUC_train_LR))
print('The loss is: %.3f' % (loss_train_LR))
print("F1 score is: " + str("{0:.2f}".format(100 * metrics.f1_score(y_train, y_pred_train_LR, average='macro'))) + "%")
print("Accuracy is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(y_train, y_pred_train_LR))) + "%")

cnf_matrix_test = metrics.confusion_matrix(y_pred_test_LR, y_test)
ax1 = plt.subplot()
sns.heatmap(cnf_matrix_test, annot=True, xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
ax1.set(ylabel='True labels', xlabel='Predicted labels')
plt.show()

print('Evaluation metrics for the test set:')
print('AUC is: %.3f' % (AUC_test_LR))
print('The loss is: %.3f' % (loss_test_LR))
print("F1 score is: " + str("{0:.2f}".format(100 * metrics.f1_score(y_test, y_pred_test_LR, average='macro'))) + "%")
print("Accuracy is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(y_test, y_pred_test_LR))) + "%")

from sklearn.model_selection import StratifiedKFold
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, random_state=10, shuffle=True)

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import hinge_loss
from sklearn.metrics import confusion_matrix
svc = SVC(probability=True)
pipe = Pipeline(steps=[('svm', svc)])
C = [0.001,0.01,0.5,1,1.5,2,3,4,5,10,20,50,100,1000]
svm_lin = GridSearchCV(estimator=pipe,
             param_grid={'svm__kernel':['linear'], 'svm__C':C},
             scoring=['accuracy','f1','precision','recall','roc_auc'],
             cv=skf, refit='roc_auc', verbose=3, return_train_score=True)
svm_lin.fit(X_train, y_train)
best_svm_lin = svm_lin.best_estimator_
print(svm_lin.best_params_)

y_pred_train_svm_lin = best_svm_lin.predict(X_train)
y_pred_proba_train_svm_lin = best_svm_lin.predict_proba(X_train)
AUC_train_svm_lin = roc_auc_score(y_train, y_pred_proba_train_svm_lin[:,1])
loss_train_svm_lin = hinge_loss(y_train, y_pred_train_svm_lin)

y_pred_test_svm_lin = best_svm_lin.predict(X_test)
y_pred_proba_test_svm_lin = best_svm_lin.predict_proba(X_test)
AUC_test_svm_lin = roc_auc_score(y_test, y_pred_proba_test_svm_lin[:,1])
loss_test_svm_lin = hinge_loss(y_test, y_pred_test_svm_lin)

cnf_matrix_train = metrics.confusion_matrix(y_pred_train_svm_lin, y_train)
ax1 = plt.subplot()
sns.heatmap(cnf_matrix_train, annot=True, xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
ax1.set(ylabel='True labels', xlabel='Predicted labels')
plt.show()

print('AUC is: %.3f' % (AUC_train_svm_lin))
print('The loss is: %.3f' % (loss_train_svm_lin))
print("F1 score is: " + str("{0:.2f}".format(100 * metrics.f1_score(y_train, y_pred_train_svm_lin, average='macro'))) + "%")
print("Accuracy is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(y_train, y_pred_train_svm_lin))) + "%")


cnf_matrix_test = metrics.confusion_matrix(y_pred_test_svm_lin, y_test)
ax1 = plt.subplot()
sns.heatmap(cnf_matrix_test, annot=True, xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
ax1.set(ylabel='True labels', xlabel='Predicted labels')
plt.show()

print('AUC is: %.3f' % (AUC_test_svm_lin))
print('The loss is: %.3f' % (loss_test_svm_lin))
print("F1 score is: " + str("{0:.2f}".format(100 * metrics.f1_score(y_test, y_pred_test_svm_lin, average='macro'))) + "%")
print("Accuracy is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(y_test, y_pred_test_svm_lin))) + "%")


C = [0.001,0.01,0.5,1,1.5,2,3,4,5,10,20,50,100,1000]
svm_nonlin = GridSearchCV(estimator=pipe,
             param_grid={'svm__kernel':['rbf','poly'], 'svm__C':C, 'svm__degree':[3], 'svm__gamma':['auto','scale']},
             scoring=['accuracy','f1','precision','recall','roc_auc'],
             cv=skf, refit='roc_auc', verbose=3, return_train_score=True)
svm_nonlin.fit(X_train, y_train)
best_svm_nonlin = svm_nonlin.best_estimator_
print(svm_nonlin.best_params_)

y_pred_train_svm_nonlin = best_svm_nonlin.predict(X_train)
y_pred_proba_train_svm_nonlin = best_svm_nonlin.predict_proba(X_train)
AUC_train_svm_nonlin = roc_auc_score(y_train, y_pred_proba_train_svm_nonlin[:,1])
loss_train_svm_nonlin = hinge_loss(y_train, y_pred_train_svm_nonlin)

y_pred_test_svm_nonlin = best_svm_nonlin.predict(X_test)
y_pred_proba_test_svm_nonlin = best_svm_nonlin.predict_proba(X_test)
AUC_test_svm_nonlin = roc_auc_score(y_test, y_pred_proba_test_svm_nonlin[:,1])
loss_test_svm_nonlin = hinge_loss(y_test, y_pred_test_svm_nonlin)


cnf_matrix_train = metrics.confusion_matrix(y_pred_train_svm_nonlin, y_train)
ax1 = plt.subplot()
sns.heatmap(cnf_matrix_train, annot=True, xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
ax1.set(ylabel='True labels', xlabel='Predicted labels')
plt.show()

print('AUC is: %.3f' % (AUC_train_svm_nonlin))
print('The loss is: %.3f' % (loss_train_svm_nonlin))
print("F1 score is: " + str("{0:.2f}".format(100 * metrics.f1_score(y_train, y_pred_train_svm_nonlin, average='macro'))) + "%")
print("Accuracy is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(y_train, y_pred_train_svm_nonlin))) + "%")


cnf_matrix_test = metrics.confusion_matrix(y_pred_test_svm_nonlin, y_test)
ax1 = plt.subplot()
sns.heatmap(cnf_matrix_test, annot=True, xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
ax1.set(ylabel='True labels', xlabel='Predicted labels')
plt.show()

print('AUC is: %.3f' % (AUC_test_svm_nonlin))
print('The loss is: %.3f' % (loss_test_svm_nonlin))
print("F1 score is: " + str("{0:.2f}".format(100 * metrics.f1_score(y_test, y_pred_test_svm_nonlin, average='macro'))) + "%")
print("Accuracy is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(y_test, y_pred_test_svm_nonlin))) + "%")


from sklearn.ensemble import RandomForestClassifier as rfc
clf = rfc(n_estimators=10)
clf.fit(X_train, y_train)

y_pred_train_rfc = clf.predict(X_train)
y_pred_proba_train_rfc = clf.predict_proba(X_train)
AUC_train_rfc = roc_auc_score(y_train, y_pred_proba_train_rfc[:,1])
loss_train_rfc = hinge_loss(y_train, y_pred_train_rfc)

y_pred_test_rfc = clf.predict(X_test)
y_pred_proba_test_rfc = clf.predict_proba(X_test)
AUC_test_rfc = roc_auc_score(y_test, y_pred_proba_test_rfc[:,1])
loss_test_rfc = hinge_loss(y_test, y_pred_test_rfc)

cnf_matrix_train = metrics.confusion_matrix(y_pred_train_rfc, y_train)
ax1 = plt.subplot()
sns.heatmap(cnf_matrix_train, annot=True, xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
ax1.set(ylabel='True labels', xlabel='Predicted labels')
plt.show()

print('AUC is: %.3f' % (AUC_train_rfc))
print('The loss is: %.3f' % (loss_train_rfc))
print("F1 score is: " + str("{0:.2f}".format(100 * metrics.f1_score(y_train, y_pred_train_rfc, average='macro'))) + "%")
print("Accuracy is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(y_train, y_pred_train_rfc))) + "%")

cnf_matrix_test = metrics.confusion_matrix(y_pred_test_rfc, y_test)
ax1 = plt.subplot()
sns.heatmap(cnf_matrix_test, annot=True, xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
ax1.set(ylabel='True labels', xlabel='Predicted labels')
plt.show()

print('AUC is: %.3f' % (AUC_test_rfc))
print('The loss is: %.3f' % (loss_test_rfc))
print("F1 score is: " + str("{0:.2f}".format(100 * metrics.f1_score(y_test, y_pred_test_rfc, average='macro'))) + "%")
print("Accuracy is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(y_test, y_pred_test_rfc))) + "%")

feature_importance = clf.feature_importances_
sns.barplot(x=col_names , y=feature_importance)
print(feature_importance)


from sklearn.decomposition import PCA
from HW2_functions import plt_2d_pca
n_components = 2
pca = PCA(n_components=n_components, whiten=True)
scaler = StandardScaler()
X_train_stdscl = scaler.fit_transform(X_train_notscl)
X_test_stdscl = scaler.transform(X_test_notscl)
X_train_pca = pca.fit_transform(X_train_stdscl)
X_test_pca = pca.transform(X_test_stdscl)

plt_2d_pca(X_train_pca[:,0:2],y_train)
plt_2d_pca(X_test_pca[:,0:2],y_test)

# LR
C = 1
penalty = 'l2'
logreg = LogisticRegression(solver='saga', multi_class='ovr', penalty=penalty, C=C, max_iter=10000)
logreg.fit(X_train_pca, y_train)

y_pred_train_pca_LR = logreg.predict(X_train_pca)
y_pred_proba_train_pca_LR = logreg.predict_proba(X_train_pca)
AUC_train_pca_LR = roc_auc_score(y_train, y_pred_proba_train_pca_LR[:,1])
loss_train_pca_LR = log_loss(y_train, y_pred_train_pca_LR)

y_pred_test_pca_LR = logreg.predict(X_test_pca)
y_pred_proba_test_pca_LR = logreg.predict_proba(X_test_pca)
AUC_test_pca_LR = roc_auc_score(y_test, y_pred_proba_test_pca_LR[:,1])
loss_test_pca_LR = log_loss(y_test, y_pred_test_pca_LR)

cnf_matrix_train = metrics.confusion_matrix(y_pred_train_pca_LR, y_train)
ax1 = plt.subplot()
sns.heatmap(cnf_matrix_train, annot=True, xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
ax1.set(ylabel='True labels', xlabel='Predicted labels')
plt.show()

print('Evaluation metrics for the training set:')
print('AUC is: %.3f' % (AUC_train_pca_LR))
print('The loss is: %.3f' % (loss_train_pca_LR))
print("F1 score is: " + str("{0:.2f}".format(100 * metrics.f1_score(y_train, y_pred_train_pca_LR, average='macro'))) + "%")
print("Accuracy is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(y_train, y_pred_train_pca_LR))) + "%")

cnf_matrix_test = metrics.confusion_matrix(y_pred_test_pca_LR, y_test)
ax1 = plt.subplot()
sns.heatmap(cnf_matrix_test, annot=True, xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
ax1.set(ylabel='True labels', xlabel='Predicted labels')
plt.show()

print('Evaluation metrics for the test set:')
print('AUC is: %.3f' % (AUC_test_pca_LR))
print('The loss is: %.3f' % (loss_test_pca_LR))
print("F1 score is: " + str("{0:.2f}".format(100 * metrics.f1_score(y_test, y_pred_test_pca_LR, average='macro'))) + "%")
print("Accuracy is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(y_test, y_pred_test_pca_LR))) + "%")


# SVM linear
svc = SVC(probability=True)
pipe = Pipeline(steps=[('svm', svc)])
C = [0.001,0.01,0.5,1,1.5,2,3,4,5,10,20,50,100,1000]
svm_lin = GridSearchCV(estimator=pipe,
             param_grid={'svm__kernel':['linear'], 'svm__C':C},
             scoring=['accuracy','f1','precision','recall','roc_auc'],
             cv=skf, refit='roc_auc', verbose=3, return_train_score=True)
svm_lin.fit(X_train_pca, y_train)
best_svm_lin = svm_lin.best_estimator_
print(svm_lin.best_params_)

y_pred_train_pca_svm_lin = best_svm_lin.predict(X_train_pca)
y_pred_proba_train_pca_svm_lin = best_svm_lin.predict_proba(X_train_pca)
AUC_train_pca_svm_lin = roc_auc_score(y_train, y_pred_proba_train_pca_svm_lin[:,1])
loss_train_pca_svm_lin = hinge_loss(y_train, y_pred_train_pca_svm_lin)

y_pred_test_pca_svm_lin = best_svm_lin.predict(X_test_pca)
y_pred_proba_test_pca_svm_lin = best_svm_lin.predict_proba(X_test_pca)
AUC_test_pca_svm_lin = roc_auc_score(y_test, y_pred_proba_test_pca_svm_lin[:,1])
loss_test_pca_svm_lin = hinge_loss(y_test, y_pred_test_pca_svm_lin)

cnf_matrix_train = metrics.confusion_matrix(y_pred_train_pca_svm_lin, y_train)
ax1 = plt.subplot()
sns.heatmap(cnf_matrix_train, annot=True, xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
ax1.set(ylabel='True labels', xlabel='Predicted labels')
plt.show()

print('AUC is: %.3f' % (AUC_train_pca_svm_lin))
print('The loss is: %.3f' % (loss_train_pca_svm_lin))
print("F1 score is: " + str("{0:.2f}".format(100 * metrics.f1_score(y_train, y_pred_train_pca_svm_lin, average='macro'))) + "%")
print("Accuracy is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(y_train, y_pred_train_pca_svm_lin))) + "%")


cnf_matrix_test = metrics.confusion_matrix(y_pred_test_pca_svm_lin, y_test)
ax1 = plt.subplot()
sns.heatmap(cnf_matrix_test, annot=True, xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
ax1.set(ylabel='True labels', xlabel='Predicted labels')
plt.show()

print('AUC is: %.3f' % (AUC_test_pca_svm_lin))
print('The loss is: %.3f' % (loss_test_pca_svm_lin))
print("F1 score is: " + str("{0:.2f}".format(100 * metrics.f1_score(y_test, y_pred_test_pca_svm_lin, average='macro'))) + "%")
print("Accuracy is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(y_test, y_pred_test_pca_svm_lin))) + "%")




C = [0.001,0.01,0.5,1,1.5,2,3,4,5,10,20,50,100,1000]
svm_nonlin = GridSearchCV(estimator=pipe,
             param_grid={'svm__kernel':['rbf','poly'], 'svm__C':C, 'svm__degree':[3], 'svm__gamma':['auto','scale']},
             scoring=['accuracy','f1','precision','recall','roc_auc'],
             cv=skf, refit='roc_auc', verbose=3, return_train_score=True)
svm_nonlin.fit(X_train_pca, y_train)
best_svm_nonlin = svm_nonlin.best_estimator_
print(svm_nonlin.best_params_)

y_pred_train_pca_svm_nonlin = best_svm_nonlin.predict(X_train_pca)
y_pred_proba_train_pca_svm_nonlin = best_svm_nonlin.predict_proba(X_train_pca)
AUC_train_pca_svm_nonlin = roc_auc_score(y_train, y_pred_proba_train_pca_svm_nonlin[:,1])
loss_train_pca_svm_nonlin = hinge_loss(y_train, y_pred_train_pca_svm_nonlin)

y_pred_test_pca_svm_nonlin = best_svm_nonlin.predict(X_test_pca)
y_pred_proba_test_pca_svm_nonlin = best_svm_nonlin.predict_proba(X_test_pca)
AUC_test_pca_svm_nonlin = roc_auc_score(y_test, y_pred_proba_test_pca_svm_nonlin[:,1])
loss_test_pca_svm_nonlin = hinge_loss(y_test, y_pred_test_pca_svm_nonlin)


cnf_matrix_train = metrics.confusion_matrix(y_pred_train_pca_svm_nonlin, y_train)
ax1 = plt.subplot()
sns.heatmap(cnf_matrix_train, annot=True, xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
ax1.set(ylabel='True labels', xlabel='Predicted labels')
plt.show()

print('AUC is: %.3f' % (AUC_train_pca_svm_nonlin))
print('The loss is: %.3f' % (loss_train_pca_svm_nonlin))
print("F1 score is: " + str("{0:.2f}".format(100 * metrics.f1_score(y_train, y_pred_train_pca_svm_nonlin, average='macro'))) + "%")
print("Accuracy is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(y_train, y_pred_train_pca_svm_nonlin))) + "%")


cnf_matrix_test = metrics.confusion_matrix(y_pred_test_pca_svm_nonlin, y_test)
ax1 = plt.subplot()
sns.heatmap(cnf_matrix_test, annot=True, xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
ax1.set(ylabel='True labels', xlabel='Predicted labels')
plt.show()

print('AUC is: %.3f' % (AUC_test_pca_svm_nonlin))
print('The loss is: %.3f' % (loss_test_pca_svm_nonlin))
print("F1 score is: " + str("{0:.2f}".format(100 * metrics.f1_score(y_test, y_pred_test_pca_svm_nonlin, average='macro'))) + "%")
print("Accuracy is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(y_test, y_pred_test_pca_svm_nonlin))) + "%")







# 2 Best feats
X_train_2fts = X_train[['Increased Urination', 'Increased Thirst']]
X_test_2fts = X_test[['Increased Urination', 'Increased Thirst']]

# LR
C = 1
penalty = 'l2'
logreg = LogisticRegression(solver='saga', multi_class='ovr', penalty=penalty, C=C, max_iter=10000)
logreg.fit(X_train_2fts, y_train)

y_pred_train_2fts_LR = logreg.predict(X_train_2fts)
y_pred_proba_train_2fts_LR = logreg.predict_proba(X_train_2fts)
AUC_train_2fts_LR = roc_auc_score(y_train, y_pred_proba_train_2fts_LR[:,1])
loss_train_2fts_LR = log_loss(y_train, y_pred_train_2fts_LR)

y_pred_test_2fts_LR = logreg.predict(X_test_2fts)
y_pred_proba_test_2fts_LR = logreg.predict_proba(X_test_2fts)
AUC_test_2fts_LR = roc_auc_score(y_test, y_pred_proba_test_2fts_LR[:,1])
loss_test_2fts_LR = log_loss(y_test, y_pred_test_2fts_LR)

cnf_matrix_train = metrics.confusion_matrix(y_pred_train_2fts_LR, y_train)
ax1 = plt.subplot()
sns.heatmap(cnf_matrix_train, annot=True, xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
ax1.set(ylabel='True labels', xlabel='Predicted labels')
plt.show()

print('Evaluation metrics for the training set:')
print('AUC is: %.3f' % (AUC_train_2fts_LR))
print('The loss is: %.3f' % (loss_train_2fts_LR))
print("F1 score is: " + str("{0:.2f}".format(100 * metrics.f1_score(y_train, y_pred_train_2fts_LR, average='macro'))) + "%")
print("Accuracy is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(y_train, y_pred_train_2fts_LR))) + "%")

cnf_matrix_test = metrics.confusion_matrix(y_pred_test_2fts_LR, y_test)
ax1 = plt.subplot()
sns.heatmap(cnf_matrix_test, annot=True, xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
ax1.set(ylabel='True labels', xlabel='Predicted labels')
plt.show()

print('Evaluation metrics for the test set:')
print('AUC is: %.3f' % (AUC_test_2fts_LR))
print('The loss is: %.3f' % (loss_test_2fts_LR))
print("F1 score is: " + str("{0:.2f}".format(100 * metrics.f1_score(y_test, y_pred_test_2fts_LR, average='macro'))) + "%")
print("Accuracy is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(y_test, y_pred_test_2fts_LR))) + "%")


# SVM linear
svc = SVC(probability=True)
pipe = Pipeline(steps=[('svm', svc)])
C = [0.001,0.01,0.5,1,1.5,2,3,4,5,10,20,50,100,1000]
svm_lin = GridSearchCV(estimator=pipe,
             param_grid={'svm__kernel':['linear'], 'svm__C':C},
             scoring=['accuracy','f1','precision','recall','roc_auc'],
             cv=skf, refit='roc_auc', verbose=3, return_train_score=True)
svm_lin.fit(X_train_2fts, y_train)
best_svm_lin = svm_lin.best_estimator_
print(svm_lin.best_params_)

y_pred_train_2fts_svm_lin = best_svm_lin.predict(X_train_2fts)
y_pred_proba_train_2fts_svm_lin = best_svm_lin.predict_proba(X_train_2fts)
AUC_train_2fts_svm_lin = roc_auc_score(y_train, y_pred_proba_train_2fts_svm_lin[:,1])
loss_train_2fts_svm_lin = hinge_loss(y_train, y_pred_train_2fts_svm_lin)

y_pred_test_2fts_svm_lin = best_svm_lin.predict(X_test_2fts)
y_pred_proba_test_2fts_svm_lin = best_svm_lin.predict_proba(X_test_2fts)
AUC_test_2fts_svm_lin = roc_auc_score(y_test, y_pred_proba_test_2fts_svm_lin[:,1])
loss_test_2fts_svm_lin = hinge_loss(y_test, y_pred_test_2fts_svm_lin)

cnf_matrix_train = metrics.confusion_matrix(y_pred_train_2fts_svm_lin, y_train)
ax1 = plt.subplot()
sns.heatmap(cnf_matrix_train, annot=True, xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
ax1.set(ylabel='True labels', xlabel='Predicted labels')
plt.show()

print('AUC is: %.3f' % (AUC_train_2fts_svm_lin))
print('The loss is: %.3f' % (loss_train_2fts_svm_lin))
print("F1 score is: " + str("{0:.2f}".format(100 * metrics.f1_score(y_train, y_pred_train_2fts_svm_lin, average='macro'))) + "%")
print("Accuracy is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(y_train, y_pred_train_2fts_svm_lin))) + "%")


cnf_matrix_test = metrics.confusion_matrix(y_pred_test_2fts_svm_lin, y_test)
ax1 = plt.subplot()
sns.heatmap(cnf_matrix_test, annot=True, xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
ax1.set(ylabel='True labels', xlabel='Predicted labels')
plt.show()

print('AUC is: %.3f' % (AUC_test_2fts_svm_lin))
print('The loss is: %.3f' % (loss_test_2fts_svm_lin))
print("F1 score is: " + str("{0:.2f}".format(100 * metrics.f1_score(y_test, y_pred_test_2fts_svm_lin, average='macro'))) + "%")
print("Accuracy is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(y_test, y_pred_test_2fts_svm_lin))) + "%")




C = [0.001,0.01,0.5,1,1.5,2,3,4,5,10,20,50,100,1000]
svm_nonlin = GridSearchCV(estimator=pipe,
             param_grid={'svm__kernel':['rbf','poly'], 'svm__C':C, 'svm__degree':[3], 'svm__gamma':['auto','scale']},
             scoring=['accuracy','f1','precision','recall','roc_auc'],
             cv=skf, refit='roc_auc', verbose=3, return_train_score=True)
svm_nonlin.fit(X_train_2fts, y_train)
best_svm_nonlin = svm_nonlin.best_estimator_
print(svm_nonlin.best_params_)

y_pred_train_2fts_svm_nonlin = best_svm_nonlin.predict(X_train_2fts)
y_pred_proba_train_2fts_svm_nonlin = best_svm_nonlin.predict_proba(X_train_2fts)
AUC_train_2fts_svm_nonlin = roc_auc_score(y_train, y_pred_proba_train_2fts_svm_nonlin[:,1])
loss_train_2fts_svm_nonlin = hinge_loss(y_train, y_pred_train_2fts_svm_nonlin)

y_pred_test_2fts_svm_nonlin = best_svm_nonlin.predict(X_test_2fts)
y_pred_proba_test_2fts_svm_nonlin = best_svm_nonlin.predict_proba(X_test_2fts)
AUC_test_2fts_svm_nonlin = roc_auc_score(y_test, y_pred_proba_test_2fts_svm_nonlin[:,1])
loss_test_2fts_svm_nonlin = hinge_loss(y_test, y_pred_test_2fts_svm_nonlin)


cnf_matrix_train = metrics.confusion_matrix(y_pred_train_2fts_svm_nonlin, y_train)
ax1 = plt.subplot()
sns.heatmap(cnf_matrix_train, annot=True, xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
ax1.set(ylabel='True labels', xlabel='Predicted labels')
plt.show()

print('AUC is: %.3f' % (AUC_train_2fts_svm_nonlin))
print('The loss is: %.3f' % (loss_train_2fts_svm_nonlin))
print("F1 score is: " + str("{0:.2f}".format(100 * metrics.f1_score(y_train, y_pred_train_2fts_svm_nonlin, average='macro'))) + "%")
print("Accuracy is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(y_train, y_pred_train_2fts_svm_nonlin))) + "%")


cnf_matrix_test = metrics.confusion_matrix(y_pred_test_2fts_svm_nonlin, y_test)
ax1 = plt.subplot()
sns.heatmap(cnf_matrix_test, annot=True, xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
ax1.set(ylabel='True labels', xlabel='Predicted labels')
plt.show()

print('AUC is: %.3f' % (AUC_test_2fts_svm_nonlin))
print('The loss is: %.3f' % (loss_test_2fts_svm_nonlin))
print("F1 score is: " + str("{0:.2f}".format(100 * metrics.f1_score(y_test, y_pred_test_2fts_svm_nonlin, average='macro'))) + "%")
print("Accuracy is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(y_test, y_pred_test_2fts_svm_nonlin))) + "%")

print('hi')

print('hi')
