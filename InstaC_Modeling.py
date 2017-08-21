#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Predicting Instacart user product reorders --
Script for model training and kaggle test set submission using
model predicted probabilities and F1 score optimization.

The code for F1 optimization was kindly provided as a kaggle kernel by Faron:
https://www.kaggle.com/mmueller/f1-score-expectation-maximization-in-o-n

Created on Sat Jul 29 14:19:02 2017

@author: josepheddy
"""

from collections import OrderedDict

import pickle
import pandas as pd
import numpy as np
from scipy import sparse

import xgboost as xgb

#==============================================================================
#  Gradient boosting model creation
#==============================================================================

X_train = pd.read_csv('df_train_full.csv') 

#cross-validation with 5% of the training data
ids = X_train['user_id'].unique()
np.random.shuffle(ids)
train_ids, test_ids = ids[:int(len(ids)*.95)], ids[int(len(ids)*.95):]
 
X_test = X_train[X_train['user_id'].isin(test_ids)]
X_train = X_train[X_train['user_id'].isin(train_ids)]         

y_train = X_train['in_order']
X_train = X_train.drop(['order_id','user_id','product_id',
                        'product_list','in_order'],axis=1)

y_test = X_test['in_order']
X_test = X_test.drop(['order_id','user_id','product_id',
                      'product_list','in_order'],axis=1)

coef_names = X_train.columns
X_train = sparse.csr_matrix(X_train)
X_test = sparse.csr_matrix(X_test)

#can cross-val on these parameters to get optimal results
gbm = xgb.XGBClassifier(max_depth=6, n_estimators=50000, 
                        learning_rate=0.05, subsample=.76, 
                        colsample_bytree=.95, nthread=-1)
eval_set = [(X_train, y_train), (X_test, y_test)]
gbm.fit(X_train, y_train, early_stopping_rounds=100, 
        eval_metric=["logloss"], eval_set=eval_set)

print('Created model')

#get feature importances
df_imps = pd.DataFrame({'feature':coef_names, 'importance':gbm.feature_importances_},
                        columns=['feature','importance'])
df_imps.sort_values(by='importance',ascending=False,inplace=True)
print(df_imps)

#save model
pickle.dump(gbm, open("gbm_model.pickle.dat", "wb"))
                             
#==============================================================================
#  Code from Faron's F1-score optimization kernel
#==============================================================================

'''
This kernel implements the O(n²) F1-Score expectation maximization algorithm presented in
"Ye, N., Chai, K., Lee, W., and Chieu, H.  Optimizing F-measures: A Tale of Two Approaches. In ICML, 2012."

It solves argmax_(0 <= k <= n,[[None]]) E[F1(P,k,[[None]])]
with [[None]] being the indicator for predicting label "None"
given posteriors P = [p_1, p_2, ... , p_n], where p_1 > p_2 > ... > p_n
under label independence assumption by means of dynamic programming in O(n²).

@author: Faron                                                                       
'''

class F1Optimizer():
    def __init__(self):
        pass

    @staticmethod
    def get_expectations(P, pNone=None):
        expectations = []
        P = np.sort(P)[::-1]

        n = np.array(P).shape[0]
        DP_C = np.zeros((n + 2, n + 1))
        if pNone is None:
            pNone = (1.0 - P).prod()

        DP_C[0][0] = 1.0
        for j in range(1, n):
            DP_C[0][j] = (1.0 - P[j - 1]) * DP_C[0, j - 1]

        for i in range(1, n + 1):
            DP_C[i, i] = DP_C[i - 1, i - 1] * P[i - 1]
            for j in range(i + 1, n + 1):
                DP_C[i, j] = P[j - 1] * DP_C[i - 1, j - 1] + (1.0 - P[j - 1]) * DP_C[i, j - 1]

        DP_S = np.zeros((2 * n + 1,))
        DP_SNone = np.zeros((2 * n + 1,))
        for i in range(1, 2 * n + 1):
            DP_S[i] = 1. / (1. * i)
            DP_SNone[i] = 1. / (1. * i + 1)
        for k in range(n + 1)[::-1]:
            f1 = 0
            f1None = 0
            for k1 in range(n + 1):
                f1 += 2 * k1 * DP_C[k1][k] * DP_S[k + k1]
                f1None += 2 * k1 * DP_C[k1][k] * DP_SNone[k + k1]
            for i in range(1, 2 * k - 1):
                DP_S[i] = (1 - P[k - 1]) * DP_S[i] + P[k - 1] * DP_S[i + 1]
                DP_SNone[i] = (1 - P[k - 1]) * DP_SNone[i] + P[k - 1] * DP_SNone[i + 1]
            expectations.append([f1None + 2 * pNone / (2 + k), f1])

        return np.array(expectations[::-1]).T

    @staticmethod
    def maximize_expectation(P, pNone=None):
        expectations = F1Optimizer.get_expectations(P, pNone)

        ix_max = np.unravel_index(expectations.argmax(), expectations.shape)
        max_f1 = expectations[ix_max]

        predNone = True if ix_max[0] == 0 else False
        best_k = ix_max[1]

        return best_k, predNone, max_f1

    @staticmethod
    def _F1(tp, fp, fn):
        return 2 * tp / (2 * tp + fp + fn)

    @staticmethod
    def _Fbeta(tp, fp, fn, beta=1.0):
        beta_squared = beta ** 2
        return (1.0 + beta_squared) * tp / ((1.0 + beta_squared) * tp + fp + beta_squared * fn)

#==============================================================================
#  Kaggle prediction submission using model probabilities and F1 optimization
#==============================================================================

X = pd.read_csv('df_test.csv')
opt = F1Optimizer()
             
df_submission = X[['order_id','product_id']]
X = X.drop(['order_id','user_id','product_id'],axis=1)
X = sparse.csr_matrix(X)

df_submission['probs_pred'] = gbm.predict_proba(X)[:,1] 

df_submission = \
df_submission.groupby(['order_id'],as_index=False) \
                       .agg(OrderedDict([('product_id',(lambda x: tuple(x))),
                                         ('probs_pred',(lambda x: tuple(x)))]))

predicted_prods = []
i = 0

#for each user feed descending order product probabilities to F1 optimizer
#to get number of products to predict (and if to predict None),
#then add the corresponding products/None to the submission  
for products, probs in zip(df_submission['product_id'],df_submission['probs_pred']):
    ordered_pairs = sorted(zip(products, probs), key=lambda pair: pair[1])[::-1] 
    products = [x for (x,y) in ordered_pairs]
    probs = [y for (x,y) in ordered_pairs]   
    best_k, predNone, _ = opt.maximize_expectation(probs)
    k_prods = products[:best_k]
    if (predNone == True):
        k_prods.append('None')
    predicted_prods.append(k_prods)
    if (i % 10000 == 0): #status checker
        print(i) 
    i += 1
df_submission['products'] = predicted_prods
 
sub = df_submission.drop(['product_id','probs_pred'],axis=1)
sub.fillna(value='None',inplace=True)
sub['products'] = sub['products'].apply(lambda x: x if x == 'None' 
                                                    else ' '.join(map(str,x)))

sub.to_csv('instacart_sub.csv',index=False)
