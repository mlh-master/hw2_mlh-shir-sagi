import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
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
    train_pos_diag = (y_train=='Positive').sum()
    Diag.append((train_pos_diag/len(y_train))*100)
    test_pos_diag = (y_test=='Positive').sum()
    Diag.append((test_pos_diag/len(y_test))*100)
    for feat in feats:
        train.append((((X_trn[feat] == 'Yes' ).sum()+(X_trn[feat] == 1).sum())/len(X_trn[feat]))*100)
        test.append((((X_tst[feat] == 'Yes' ).sum()+(X_tst[feat] == 1).sum())/len(X_tst[feat]))*100)

    # create plot
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
    del X_trn[remove_feat]
    feats = X_trn.columns
    count = 1
    for feat in feats:
        f_n_d_n = 0
        f_n_d_p = 0
        f_y_d_n = 0
        f_y_d_p = 0
        for idx, v in enumerate(X_trn[feat]):
            a = v
            b = y_train[idx]
            if (v == 'No' or v == 0 or v == 'Female') and y_train[idx] == 'Negative':
                f_n_d_n = f_n_d_n +1
            elif (v == 'No' or v == 0 or v == 'Female') and y_train[idx] == 'Positive':
                f_n_d_p = f_n_d_p + 1
            elif (v == 'Yes' or v == 1 or v == 'Male') and y_train[idx] == 'Negative':
                f_y_d_n = f_y_d_n + 1
            elif (v == 'Yes' or v == 1 or v == 'Male') and y_train[idx] == 'Positive':
                f_y_d_p = f_y_d_p + 1
        feat_no = []
        feat_yes = []
        feat_no.append(f_n_d_n)
        feat_no.append(f_n_d_p)
        feat_yes.append(f_y_d_n)
        feat_yes.append(f_y_d_p)
        index = np.arange(2)
        bar_width = 0.35
        opacity = 0.8
        ax1 = plt.subplot(4416)
        # plt.bar(['Train', 'Test'], feat_no, bar_width, color=['steelblue', 'lightskyblue'])
        rects1 = plt.bar(index, feat_no, bar_width, alpha=opacity, color='steelblue', label='Train set')
        rects2 = plt.bar(index + bar_width, feat_yes, bar_width, alpha=opacity, color='lightskyblue', label='Test set')
        plt.xlabel('Features')
        plt.ylabel('Count')
        count = count + 1
        plt.show()

        x = np.arange(11)
        y = np.random.rand(len(x), 9) * 10

        fig, axes = plt.subplots(3, 3, sharex=True, sharey=True)

        for i, ax in enumerate(axes.flatten()):
            ax.bar(x, y[:, i], color=plt.cm.Paired(i / 10.))

        plt.show()


    # ax1 = plt.subplot(441)
    # rects1 = plt.bar(index, train, bar_width,alpha=opacity,color='steelblue',label='Train set')
    # rects2 = plt.bar(index + bar_width, test, bar_width,alpha=opacity,color='lightskyblue',label='Test set')
    # plt.hist(nsd_res[x], bins, )
    # ax1.set(ylabel='Count', xlabel='Value', title='Feature 1 scaled')
    # ax2 = plt.subplot(442)
    # plt.hist(nsd_res[y], bins, color='orange')
    # ax2.set(ylabel='Count', xlabel='Value', title='Feature 2 scaled')
    # ax3 = plt.subplot(443)
    # plt.hist(CTG_features[x], bins)
    # ax3.set(ylabel='Count', xlabel='Value', title='Feature 1 unscaled')
    # ax4 = plt.subplot(444)
    # plt.hist(CTG_features[y], bins, color='orange')
    # ax4.set(ylabel='Count', xlabel='Value', title='Feature 2 unscaled')
    # plt.show()
    return