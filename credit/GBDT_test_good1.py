# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import recall_score, precision_score
 
#导入随机森林算法库
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle 
from sklearn.ensemble import GradientBoostingClassifier
  
df_All = pd.read_csv("train.csv", sep=',')
df_All = df_All[(df_All["label"]==0) | (df_All["label"]==1)]

df_All = df_All.fillna(-1)


df_All = shuffle(df_All) 


df_X = df_All.drop( ["certid","label"], axis=1,inplace=False)


df_y = df_All["label"]

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2)


#n_estimators树的数量一般大一点。 max_features 对于分类的话一般特征束的sqrt，auto自动
clf = GradientBoostingClassifier(learning_rate=0.01, n_estimators=500,max_depth=50, min_samples_leaf =20, min_samples_split =2, max_features="auto", subsample=0.8, random_state=10)
#clf = GradientBoostingClassifier(random_state=100, n_estimators=100)
clf = clf.fit(X_train, y_train)
  
#print clf.score(X_test, y_test)

pred = clf.predict(X_test) 

confusion_matrix=confusion_matrix(y_test,pred)
print  confusion_matrix


precision_p = float(confusion_matrix[0][0])/float((confusion_matrix[0][0] + confusion_matrix[0][1]))
recall_p = float(confusion_matrix[0][0])/float((confusion_matrix[0][0] + confusion_matrix[1][0]))
F1_Score = 2*precision_p*recall_p/(precision_p+recall_p)

print ("Precision:", precision_p) 
print ("Recall:", recall_p)
print ("F1_Score:", F1_Score)


print df_All.feature_names


FE_ip = clf.feature_importances_
print FE_ip

