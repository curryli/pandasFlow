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

Xd = pd.read_csv('1449.csv', index_col=0)
yd = pd.read_csv('train_result1449.csv', index_col=0)

X = Xd.values[:, :]
y = yd.values[:, :].ravel()

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=15)


tuned_parameters = [{ 'learning_rate' :range(0.15,0.1,0.05),
                    'n_estimators' :range(200,500,1000),
                    'max_depth' : range(4,6,8),
                    'min_child_weight' :[1,3,5,7],
                    'gamma':[i/100.0 for i in range(0,5)],
                    'subsample' : [0.4,0.6,0.8],
                    'colsample_bytree' : [0.6,0.8,0.9],
                    'objective' : 'binary:logistic',
                     'reg_alpha':[0.001,0.01,0.1,1,10],
                     'reg_lambda':[0.1,1,5],
                    'scale_pos_weight' : 1,
                    'seed' : 27}
                    ]


# scores = ['precision_macro', 'recall_macro','f1_macro'，'roc_auc']
#scores = ['f1_macro']

def my_custom_loss_func(y_test, pred):
    result = precision_recall_fscore_support(y_test, pred)
    f1_0 = result[2][0]
    return f1_0

my_score = make_scorer(my_custom_loss_func, greater_is_better=True)

scores = [my_score]


# print(gsearch1.grid_scores_,gsearch1.best_params_, gsearch1.best_score_ )

for score in scores:
    with open('XGBoost_grid.txt', 'w') as f:
        print >>f, "# Tuning hyper-parameters for %s\n" % score

        clf = GridSearchCV(XGBClassifier(scale_pos_weight=1), tuned_parameters, cv=5, scoring='%s' % score)
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

        print >>f, classification_report(y_true, y_pred)

endtime = datetime.datetime.now()
print ((endtime - starttime).seconds)          
print("over")


