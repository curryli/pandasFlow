# -*- coding: utf-8 -*-
import pandas as pd
import datetime

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from xgboost.sklearn import XGBClassifier
from sklearn.metrics import make_scorer
import numpy as np
from sklearn.metrics import precision_recall_fscore_support



starttime = datetime.datetime.now()

df_All = pd.read_csv("train_shuffled.csv", sep=',')

df_All = df_All[(df_All["label"]==0) | (df_All["label"]==1)]


df_X = df_All.drop( ["certid","label"], axis=1,inplace=False)


df_y = df_All["label"]

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2)


# tuned_parameters = [{ 'learning_rate' :range(0.15,0.1,0.05),
#                     'n_estimators' :range(200,500,1000),
#                     'max_depth' : range(4,6,8),
#                     'min_child_weight' :[1,3,5,7],
#                     'gamma':[i/100.0 for i in range(0,5)],
#                     'subsample' : [0.4,0.6,0.8],
#                     'colsample_bytree' : [0.6,0.8,0.9],
#                     'objective' : 'binary:logistic',
#                      'reg_alpha':[0.001,0.01,0.1,1,10],
#                      'reg_lambda':[0.1,1,5],
#                     'scale_pos_weight' : 1,
#                     'seed' : 27}
#                     ]
 
tuned_parameters = [{ 'learning_rate' :[0.15,0.1,0.05],
                    'n_estimators' :[200,500,1000],
                    'max_depth' : [4,5,8],
                    'min_child_weight' :[1,3,5],
                    'gamma':[0.01,0.05,0.1],
                    'subsample' : [0.6, 0.8, 0.9],
                    'colsample_bytree' : [0.7, 0.8, 0.9],
                     'reg_alpha':[0.01, 0.1, 1, 10],   #L1正则参数
                     'reg_lambda':[0.01, 0.1, 1, 10]   #L2正则参数
                    }
                    ]


def my_custom_loss_func(y_test, pred):
    result = precision_recall_fscore_support(y_test, pred)
    f1_0 = result[2][0]
    return f1_0

my_score = make_scorer(my_custom_loss_func, greater_is_better=True)



# print(gsearch1.grid_scores_,gsearch1.best_params_, gsearch1.best_score_ )


with open('XGBoost_grid.txt', 'w') as f:
    print >>f, "# Tuning hyper-parameters:\n"

    clf = GridSearchCV(XGBClassifier(objective='binary:logistic',scale_pos_weight= 1, seed=27), tuned_parameters, cv=5, scoring=my_score)
    clf.fit(X_train, y_train)

    print >>f, "Best parameters set found on development set:\n"
    print >>f, "%s" % clf.best_params_
    print >>f, "Grid scores on development set:\n"
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print >>f, "%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params)

    print >>f, "Detailed classification report:\n"

    pred = clf.predict(X_test)

    result = precision_recall_fscore_support(y_test, pred)
    precision_0 = result[0][0]
    recall_0 = result[1][0]
    f1_0 = result[2][0]
    detail_str = "precision_0: ", precision_0,"  recall_0: ", recall_0, "  f1_0: ", f1_0
    print >>f, detail_str

endtime = datetime.datetime.now()
print ((endtime - starttime).seconds)          
print("over")


