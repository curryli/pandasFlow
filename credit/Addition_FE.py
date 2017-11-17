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
import datetime
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

df_All = pd.read_csv("agg_cat.csv", sep=',')
#df_All = pd.read_csv("train_notest.csv", sep=',')
#df_All = pd.read_csv("train_1109_xyk.csv", sep=',')



# df_All_stat_0 = pd.read_csv("agg_cert.csv", sep=',')
# df_All = pd.merge(left=df_All, right=df_All_stat_0, how='left', left_on='certid', right_on='certid')

df_All_stat_0 = pd.read_csv("agg_math.csv", sep=',')
df_All = pd.merge(left=df_All, right=df_All_stat_0, how='left', left_on='certid', right_on='certid')

#df_All = df_All[(df_All["label"]==0) | (df_All["label"]==1)]

df_All_stat = pd.read_csv("translabel_stat.csv", sep=',')
df_All = pd.merge(left=df_All, right=df_All_stat, how='left', left_on='certid', right_on='certid')

df_All_stat_2 = pd.read_csv("count_label.csv", sep=',')
df_All = pd.merge(left=df_All, right=df_All_stat_2, how='left', left_on='certid', right_on='certid')

df_All_stat_3 = pd.read_csv("count_label_isnot.csv", sep=',')
df_All = pd.merge(left=df_All, right=df_All_stat_3, how='left', left_on='certid', right_on='certid')

df_All_stat_4 = pd.read_csv("groupstat_2.csv", sep=',')
df_All = pd.merge(left=df_All, right=df_All_stat_4, how='left', left_on='certid', right_on='certid')



HR_aera_1 = [2514,2601,2624,2612,2488,2518,2476,2628,1555,2487,2611,2608,1479,2474,659,2957,2713,2606,609,2892,2615,2627,870,2023,1857,2570,2028,2894,2513,2875,2009,2604,2921,2559,312,2660,933,1478,2602,2573,658,2013,2572,2998,1917,2580,2603,2954,847,2509,1652,2605,2569,2584,2494,2930,326,2485,2969,2477,610,2662,2489,2856,848,1477,35,327,2486,1881,2956,2719,2898,2607,2571,2574,484,328,266,3042,423,1916,2955,1439,2022,438,1654,1880,2834,655,384,2466,2893,2880,2925,2483,849,2473,33,2609,2492,710,1878,2631,2613,701,721,2482,1333,205,64,2970,2029,611,567,2943,758,2610,2923,2928,3110,2999,1873,2597,2661,2821,2599,656,99,2855,2853,3136,1558,2475,1444,214,415,2600,2583,1879,2889,1559,909,555,1596,2495,2590,2676,2014,3108,3012,2575,2491,1254,372,284,2024,1875,2852,216,2950,2508,221,774,2586,233,220,534,2589,429,2890,229,252,852,3065,2179,224,2515,2577,3087,225,3032,547,515,1249,2681,2598,2581,2566,13,3109,3083,3147,2634,244,2926,657,398,773,794,718,34,2941,1595,2568,2935,1332,2753,540,681,2493,2945,387,1557,2593,1556,2578,223,2931,2967,267,2614,3146,2927,2629,2846,2582,150,2588,2678,2720,2850,50,2165,2920,31,2579,497,2924,231,2696,1913,2490,771,2591,699,2764,2648,3149,2576,2680,2294,36,2587,51,242,928,1827,2966,15,1902,2484,501,122,1653,923,755,364,232,]
def is_HR_aera_1(x):
    if(x in HR_aera_1):
        return 1
    else:
        return 0
df_All["HR_aera_1"] = df_All["aera_code_encode"].map(lambda x: is_HR_aera_1(x))

HR_aera_2 = [2616,776,2724,2560,325,106,2510,53,869,319,2625,1856,109,2899,2688,873,625,647,2862,166,711,2725,553,759,2517,338,1564,529,137,315,2008,2675,2996,2069,2940,2561,731,818,379,2326,313,2623,287,468,2745,439,793,3000,3064,360,264,594,3107,2975,446,549,273,305,828,277,105,733,2871,678,424,968,107,530,2918,84,147,2734,3077,2656,378,767,450,704,2961,52,726,915,1912,2686,2844,930,1741,842,3122,371,508,2738,3002,1594,613,422,644,329,87,622,311,113,1838,473,675,717,727,203,2021,2690,2832,456,520,527,606,574,156,144,477,2888,3124,2325,190,1565,522,45,95,2622,472,2663,2685,457,437,321,2705,599,3048,459,2511,627,1740,451,3009,2636,2670,519]
def is_HR_aera_2(x):
    if(x in HR_aera_2):
        return 1
    else:
        return 0
df_All["HR_aera_2"] = df_All["aera_code_encode"].map(lambda x: is_HR_aera_2(x))


LR_app_date = [457,452,506,467,507,468,450,444,443,451,493,508,446,476,497,495,498,496,492,489,494,483,491,490,146,141,144,217,152,140,481,211,221,237,139,174,147,149,182,154,143,136,215,179,173,161,204,196,228,197,205,148,172,155,226,231,151,234,142,153,216,229,484,225,212,166,181,208,213,145,150,127,175,203,189,485,206,201,220,160,137,165,193,128,180,187,223,167,171,235,214,183,163,227,222,138,202,184,157,133,243,233,188,169,191,236,159,199,177,218,207,210,178,232,224,156,134,176,219,230,509,198,135,241,162,209,158,168,190,200,170,185,24,164,132,192,129,22,27,54,240,194,239,195,131,111,]
def is_LR_app_date(x):
    if(x in LR_app_date):
        return 1
    else:
        return 0
df_All["LR_app_date"] = df_All["apply_dateNo"].map(lambda x: is_LR_app_date(x))


LR_city = [42,53,41,17,18,40]
HR_city = [37,38,39,34,36,33,35,31]
def is_LH_city(x):
    if(x in LR_city ):
        return 2
    elif (x in HR_city):
        return 1
    else:
        return 0
df_All["LH_city"] = df_All["city"].map(lambda x: is_LH_city(x))



LR_county =[86,88,42,20,41,17,39,16,87,40]
def is_LR_county(x):
    if(x in LR_county):
        return 1
    else:
        return 0
df_All["LR_county"] = df_All["county"].map(lambda x: is_LR_county(x))


def is_NN(x):
    if((x["date-countDistinct"]==-1) & (x["date-min"]==-1) & (x["date-most_frequent_cnt"]==-1)):
        return 1
    else:
        return 0

df_All["is_NN"] = df_All.apply(is_NN, axis=1)
# date-countDistinct, date-min, date-most_frequent_cnt, date-most_frequent_item
# LR =[-1]


LR_date_cd = [234,120,247,269,340,174,288,301,320,344,196,284,325,228,289,316,261,216,179,321,211,253,238,353,397,221,293,243,312,206,292,233,248,270,220,302,229,252,197,329,285,317,188,256,225,339,193,237,266,205,298,264,161,279,180,191,286,291,281,204,259,219,274,230,245,271,240,223,306,299,267,241,255,209,194,295,282,-1,310,182,250,231,218,368,304,258,290,235,246,207,272,214,287,300,422,273,210,239,294,268,283,200,277,254,227,337,309,215,222,232]
HR_date_cd = range(10)
def is_LH_date_cd(x):
    if(x in LR_date_cd ):
        return 2
    elif (x in HR_date_cd):
        return 1
    else:
        return 0
df_All["LH_date_cd"] = df_All["dateNo-countDistinct"].map(lambda x: is_LH_date_cd(x))



LR_date_max = [69,479,88,170,115,56,142,153,42,37,25,52,504,110,157,20,46,93,57,492,179,485,121,84,147,61,132,89,116,507,60,117,102,28,160,21,137,165,484,493,53,169,499,109,471,488,124,193,489,129,503,73,105,32,34,148,45,161,149,22,71,54,144,466,181,113,76,39,98,494,103,91,483,66,155,108,130,135,505,167,35,490,162,487,458,145,48,63,18,150,-1,95,67,177,182,16,470,31,446,43,465,99,87,482,26,158,55,114,171,139,23,75,119,478,82,36,168,146,30,51,107,126,136,506,94,131,47,163,68,62,501,486,491,496,100]
HR_date_max = [72,49,104,106,96,44,86,80,58,19,90,83,78,92,40,224,122,206,230,29,38,70,33,77,41,27,112,277,296,280,238,214,268,190,284]
def is_LH_date_max(x):
    if(x in LR_date_max ):
        return 2
    elif (x in HR_date_max):
        return 1
    else:
        return 0
df_All["LH_date_max"] = df_All["dateNo-max"].map(lambda x: is_LH_date_max(x))



LR_date_min = [186,342,449,379,440,385,500,404,436,460,448,443,480,453,428,452,380,366,461,456,388,499,471,489,393,457,425,430,444,402,413,445,466,451,434,477,494,462,399,438,431,473,426,-1,414,409,370,463,470,497,465,450,482,351,390,478,383,447,432,410,506,437,400,464,459,496,454,69,88,56,89,35,67,16,99,87,55,23,82,36,107,68]
HR_date_min = [416,422,44,86,80,58,19,90,83,389,421,251,417,338,346,322,339,391,321,291,299,355,309,320,303,315,214,219,468,202,361,324,311,423,382,335,433,415,395,332,427]
def is_LH_date_min(x):
    if(x in LR_date_min ):
        return 2
    elif (x in HR_date_min):
        return 1
    else:
        return 0
df_All["LH_date_min"] = df_All["dateNo-min"].map(lambda x: is_LH_date_min(x))




LR_date_freq = [-1,447,455,439,456,444,436,451,458,450,445,442,452,462,474,461,437,468,497,469,480,476,481,466,496,475,492,483,465,495,500,487,478,491,484,477,494,490,464,479,485,467,489,499,502,504,488,503,506,501,486,505]
def is_LR_date_freq(x):
    if(x in LR_date_freq):
        return 1
    else:
        return 0
df_All["LR_date_freq"] = df_All["dateNo-most_frequent_item"].map(lambda x: is_LR_date_freq(x))




HR_hour_cd = [20,19,18,17,16,15,14,13]
def is_HR_hour_cd(x):
    if(x in HR_hour_cd):
        return 1
    else:
        return 0
df_All["HR_hour_cd"] = df_All["hour-countDistinct"].map(lambda x: is_HR_hour_cd(x))


def more_than(x):
    if(x>100):
        return 1
    else:
        return 0

df_All["LR_hour_mc"] = df_All["hour-most_frequent_cnt"].map(lambda x: more_than(x))
df_All["LR_weekday_mc"] = df_All["weekday-most_frequent_cnt"].map(lambda x: more_than(x))


LR_MA_delta = [442,479,468,481,449,440,472,492,443,485,480,453,452,429,467,484,461,493,488,489,457,425,430,423,445,466,451,434,477,483,455,490,426,487,458,-1,463,465,482,433,478,474,464,459,486,469,491,454]
def is_LR_MA_delta(x):
    if(x in LR_MA_delta):
        return 1
    else:
        return 0
df_All["LR_MA_delta"] = df_All["min_apply_delta"].map(lambda x: is_LR_MA_delta(x))


LR_prov = [37,45,42,35,36,32,33]
HR_prov =  [13,14,52,21,61,51]
def is_LH_prov(x):
    if(x in LR_prov):
        return 2
    elif (x in HR_prov):
        return 1
    else:
        return 0
df_All["LH_prov"] = df_All["prov"].map(lambda x: is_LH_prov(x))

 ####################################

df_All = df_All.fillna(-1)

df_All = shuffle(df_All)

save_lst = ["certid","label","HR_aera_1","HR_aera_2","LR_app_date","LH_city","LR_county","is_NN","LH_date_cd","LH_date_max","LH_date_min","LR_date_freq","HR_hour_cd","LR_hour_mc","LR_weekday_mc","LR_MA_delta","LH_prov"]
df_All[save_lst].to_csv("addition_stat.csv",index=False)
##########################################################################


# df_X = df_All.drop( ["certid","label"], axis=1,inplace=False)
# print df_X.columns
#
# df_y = df_All["label"]
#
# X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2)
# X_cols = X_train.columns
# sc = StandardScaler()    #MinMaxScaler()    不好
#
# #print X_train.loc[:1]
#
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
#
# #clf = XGBClassifier(learning_rate =0.1,n_estimators=500,max_depth=5,gamma=0.05,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic', reg_lambda=1,seed=27)
# clf = XGBClassifier(learning_rate =0.1,n_estimators=1000,max_depth=5,gamma=0.01,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic', reg_alpha=0.1, reg_lambda=0.1,seed=27)
#
# clf.fit(X_train, y_train)
#
# pred = clf.predict(X_test)
#
# cm1=confusion_matrix(y_test,pred)
# print  cm1
#
# precision_p = float(cm1[0][0])/float((cm1[0][0] + cm1[1][0]))
# recall_p = float(cm1[0][0])/float((cm1[0][0] + cm1[0][1]))
# F1_Score = 2*precision_p*recall_p/(precision_p+recall_p)
#
# print ("Precision:", precision_p)
# print ("Recall:", recall_p)
# print ("F1_Score:", F1_Score)
#
# FE_ip_tuples = zip(X_cols, clf.feature_importances_)
# pd.DataFrame(FE_ip_tuples).to_csv("FE_IP_xgboost_add.csv",index=True)
#
#
# #Compute precision, recall, F-measure and support for each class
# # print "weighted\n"
# # print precision_recall_fscore_support(y_test,pred, average='weighted')
#
# print "Each class\n"
# result = precision_recall_fscore_support(y_test,pred)
# #print result
# precision_0 = result[0][0]
# recall_0 = result[1][0]
# f1_0 = result[2][0]
# precision_1 = result[0][1]
# recall_1 = result[1][1]
# f1_1 = result[2][1]
# print "precision_0: ", precision_0,"  recall_0: ", recall_0, "  f1_0: ", f1_0

