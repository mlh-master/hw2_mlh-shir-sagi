import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
from sklearn.model_selection import StratifiedKFold as SKFold
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import random

def rm_nan_pat(dataframe):
    """

    :param dataframe: Pandas series of features
    :return: dataframe filtered - patients with mote than 1 missing feature removed, patients with 1 missing feature filled.
    """
    df = dataframe.copy()
    num_of_nan = df.isnull().sum(axis=1).tolist()
    rm_rows = []
    fil_rows = []
    for idx ,nan_num in enumerate(num_of_nan):
        if nan_num>1:
            rm_rows.append(idx)
        elif nan_num==1:
            fil_rows.append(idx)
    for idx in reversed(rm_rows):
            df = df[df.index != idx]
    temp_dict = df.to_dict('list')
    for k , v in temp_dict.items():
        v_clean = [x for x in v if pd.notnull(x)]
        for idx, elem in enumerate(v):
            if pd.notnull(elem)== False:
                v[idx] = np.random.choice(v_clean)
            else:
                v[idx]=elem
    return pd.DataFrame(temp_dict)

def col_charts(X_train,X_test,remove_feat, y_train, y_test):
    """

    :param X_train: Dataframe of features of all train set patients
    :param X_test: Dataframe of features of all test set patients
    :param remove_feat: Features to emit from plots
    :param y_train: Diagnosis of all train set patients
    :param y_test: Diagnosis of all test set patients
    :output: 1. Bar plot for all features of % of positive feature for both train and test set
             2. Bar plot of % of positive diagnosis for train and test set
    """
    X_trn = X_train.copy()
    X_tst = X_test.copy()
    for idx in remove_feat:
        del X_trn[idx]
        del X_tst[idx]
    feats = X_trn.columns
    n_groups = len(feats)
    train = []
    test = []
    Diag = []
    train_pos_diag = (y_train==1).sum()
    Diag.append((train_pos_diag/len(y_train))*100)
    test_pos_diag = (y_test==1).sum()
    Diag.append((test_pos_diag/len(y_test))*100)
    for feat in feats:
        train.append((((X_trn[feat] == 1).sum())/len(X_trn[feat]))*100)
        test.append((((X_tst[feat] == 1).sum())/len(X_tst[feat]))*100)

    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8
    rects1 = plt.bar(index, train, bar_width,alpha=opacity,color='steelblue',label='Train set')
    rects2 = plt.bar(index + bar_width, test, bar_width,alpha=opacity,color='lightskyblue',label='Test set')
    plt.xlabel('Features')
    plt.ylabel('Positive %')
    plt.title('% of positive feature')
    plt.ylim((1, 100))
    plt.xticks(index + bar_width, range(1,len(feats)+1))
    plt.legend()
    plt.tight_layout()
    plt.show()

    bar_width = 0.5
    plt.bar(['Train', 'Test'], Diag,bar_width,color=['steelblue', 'lightskyblue'])
    plt.ylabel('Positive %')
    plt.ylim((1, 100))
    plt.show()
    return

def feat_lable(X_train,remove_feat, y_train):
    """

    :param X_train: Dataframe of features of all train set patients
    :param X_test: Dataframe of features of all test set patients
    :param remove_feat: Features to emit from plots
    :param y_train: Diagnosis of all train set patients
    :param y_test: Diagnosis of all test set patients
    :output: 1. Bar plot for all features of % of positive feature for both train and test set
             2. Bar plot of % of positive diagnosis for train and test set
    """
    X_trn = X_train.copy()
    non_binary_feat = remove_feat
    del X_trn[remove_feat]
    feats = X_trn.columns
    f_n_d_n = []
    f_n_d_p = []
    f_y_d_n = []
    f_y_d_p = []
    for feat in feats:
        f_n_d_n_count = 0
        f_n_d_p_count = 0
        f_y_d_n_count = 0
        f_y_d_p_count = 0
        for idx, v in enumerate(X_trn[feat]):
            a = v
            b = y_train[idx]
            if v == 0 and y_train[idx] == 0:
                f_n_d_n_count = f_n_d_n_count +1
            elif v == 0 and y_train[idx] == 1:
                f_n_d_p_count = f_n_d_p_count + 1
            elif v == 1 and y_train[idx] == 0:
                f_y_d_n_count = f_y_d_n_count + 1
            elif v == 1 and y_train[idx] == 1:
                f_y_d_p_count = f_y_d_p_count + 1
        f_n_d_n.append(f_n_d_n_count)
        f_n_d_p.append(f_n_d_p_count)
        f_y_d_n.append(f_y_d_n_count)
        f_y_d_p.append(f_y_d_p_count)
    # mat = np.row_stack((f_n_d_n, f_n_d_p,f_y_d_n,f_y_d_p))
    index = np.arange(2)
    bar_width = 0.35
    opacity = 0.8
    fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(15,15))
    for i, ax in enumerate(axes.flatten()):
        rects1 = ax.bar(index, [f_n_d_n[i],f_y_d_n[i]], bar_width, alpha=opacity, color='steelblue', label='Negative')
        rects2 = ax.bar(index + bar_width, [f_n_d_p[i],f_y_d_p[i]], bar_width, alpha=opacity, color='lightskyblue', label='Positive')
        if i ==1:
            labels = ['Female', 'Male']
        else:
            labels = ['No', 'Yes']
        x = np.arange(len(labels))
        ax.set_ylabel('Count')
        if i==0:
            ax.set_xlabel(feats[i]+'(Male)')
        else:
            ax.set_xlabel(feats[i])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(prop={"size": 8}, loc='upper right')
    plt.show()

    # fig, ax = plt.subplots()
    # index = np.arange(16,91)
    # neg_diag = [i for i, x in enumerate(y_train) if x == 0]
    # pos_diag = [i for i, x in enumerate(y_train) if x == 1]
    #
    # bins = 100
    # feat = 'Width'
    # plt.hist(X_train['Age'].loc[neg_diag], bins, density=True, alpha=0.5, label='Normal')
    # plt.hist(X_train[feat].loc[pos_diag], bins, density=True, alpha=0.5, label='Suspect')
    # plt.xlabel('Histigram Width')
    # plt.ylabel('Probabilty')
    # plt.legend(loc='upper right')
    # plt.show()
    #
    #
    # X_train_neg = X_train.index.isin(neg_diag)
    # X_train_pos = X_train.index.isin(pos_diag)
    # Ages_count = Counter(X_train['Age'])
    # Ages = list(Ages_count)
    # Ages = sorted(Ages)
    # Ages_pos = []
    # Ages_neg = []
    # for Age in Ages:
    #     count_pos = 0
    #     count_neg = 0
    #     for idx in X_train['Age']:
    #         if idx==Age and y_train[idx]==0:
    #             count_neg = count_neg+1
    #         elif idx==Age and y_train[idx]==1:
    #             count_pos = count_pos+1
    #     Ages_neg.append(count_neg)
    #     Ages_pos.append(count_pos)

    #
    # bar_width = 0.35
    # opacity = 0.8
    # rects1 = plt.bar(index, train, bar_width,alpha=opacity,color='steelblue',label='Train set')
    # rects2 = plt.bar(index + bar_width, test, bar_width,alpha=opacity,color='lightskyblue',label='Test set')
    # plt.xlabel('Features')
    # plt.ylabel('Positive %')
    # plt.title('% of positive feature')
    # plt.yim((1l, 100))
    # plt.xticks(index + bar_width, range(1,len(feats)+1))
    # plt.legend()
    # plt.tight_layout()
    # plt.show()




    # bins = 100
    # feat1 = X_train[[non_binary_feat]]
    # idx_1 = (y_train == 0).index[(y_train == 0)['Diagnoisis'] == True].tolist()
    # idx_2 = (y_train == 1).index[(y_train == 1)['Diagnoisis'] == True].tolist()
    # plt.hist(feat1.loc[idx_1], bins, density=True, alpha=0.5, label='Negative')
    # plt.hist(feat1.loc[idx_2], bins, density=True, alpha=0.5, label='Positive')
    # plt.xlabel('Histigram Width')
    # plt.ylabel('Probabilty')
    # plt.legend(loc='upper right')
    # plt.show()


    return


def cv_kfold(X, y, C, penalty, K):
    """

    :param X: Training set samples
    :param y: Training set labels
    :param C: A list of regularization parameters
    :param penalty: A list of types of norm
    :param K: Number of folds
    :param mode: Mode of normalization (parameter of norm_standard function in clean_data module)
    :return: A dictionary as explained in the notebook
    """
    random.seed(10)
    kf = SKFold(n_splits=K)
    validation_dict = []
    for c in C:
        for p in penalty:
            logreg = LogisticRegression(solver='saga', penalty=p, C=c, max_iter=10000, multi_class='ovr')
            AUC_vec = np.zeros(K)
            k = 0
            for train_idx, val_idx in kf.split(X, y):
                x_train, x_val = X.iloc[train_idx], X.iloc[val_idx]
                # ------------------ IMPLEMENT YOUR CODE HERE:-----------------------------
                y_train, y_val = y[train_idx], y[val_idx]
                logreg.fit(x_train, y_train)
                y_pred = logreg.predict(x_val)
                y_pred_proba= logreg.predict_proba(x_val)
                AUC_vec[k] = roc_auc_score(y_val, y_pred_proba[:,1])
                k = k + 1
            validation_dict.append({'C': c, 'penalty': p, 'AUC': np.mean(AUC_vec)})

        # --------------------------------------------------------------------------
    return validation_dict