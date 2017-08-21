#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Predicting Instacart user product reorders --
Script for data preprocessing and feature extraction
Used to create train and test df / csvs to be used for model creation

Created on Sat Jul 29 11:48:56 2017

@author: josepheddy
"""

from collections import OrderedDict

import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr 

'''Base DataFrame Setup
'''
df_products = pd.read_csv('products.csv')
df_departments = pd.read_csv('departments.csv')
df_aisles = pd.read_csv('aisles.csv')

df_orders = pd.read_csv('orders.csv')
df_order_products_prior = pd.read_csv('order_products__prior.csv') 

df_products = pd.merge(df_products, df_aisles, on='aisle_id', how='left')
df_products = pd.merge(df_products, df_departments, on='department_id', how='left')
df_products = df_products.drop(['product_name','aisle_id','department_id'],axis=1)

df_products = pd.get_dummies(df_products)

df_order_products_train = pd.read_csv('order_products__train.csv')

df_train = \
    df_order_products_train.merge(df_orders[['order_id','user_id','order_number', \
                                             'order_dow','order_hour_of_day',
                                             'days_since_prior_order']], \
                                             on='order_id', how='left')
                       
df_order_products_prior = \
    df_order_products_prior.merge(df_orders[['order_id','user_id','order_number', \
                                             'order_dow','order_hour_of_day',
                                             'days_since_prior_order']], \
                                             on='order_id', how='left')
                        
'''Create helper dfs to extract features on user level order history, 
   user-product level order history, and generic order history  
'''
df_past_orders = df_orders[df_orders['eval_set'] == 'prior']
df_user_details = \
    df_past_orders.groupby(['user_id'],as_index=False) \
             .agg(OrderedDict([('order_number','max'),
                               ('days_since_prior_order','mean')])) \
             .rename(columns={'order_number': 'total_orders', 
                              'days_since_prior_order': 'avg_days_since_prior_order'})

df_prod_user_details = \
    df_order_products_prior.groupby(['product_id','user_id'],as_index=False) \
                           .agg(OrderedDict([('order_id','count'),
                                 ('order_number',(lambda x: tuple(x))),
                                 ('order_dow','mean'),
                                 ('order_hour_of_day','mean')]))
df_prod_user_details.columns = ['product_id','user_id','times_ordered', 
                               'orders_list','avg_dow','avg_hod']  

df_prod_user_details = pd.merge(df_prod_user_details, df_user_details, 
                                how='left', on='user_id') 
df_prod_user_details['order_freq'] = \
    df_prod_user_details['times_ordered'] / df_prod_user_details['total_orders']
df_prod_user_details['order_freq'] = np.log(df_prod_user_details['order_freq'])  
                                                        
df_order_cart_counts = \
    df_order_products_prior.groupby(['order_number','user_id'],as_index=False) \
                                    ['add_to_cart_order'].agg({'total_cart': 'max'})

#helper function to extract an ordered list of cart size of each order
def cartSize_list(user_group):
    order_nums = user_group['order_number']
    total_carts = user_group['total_cart']
    ordered_pairs = sorted(zip(order_nums, total_carts), key=lambda pair: pair[0])
    return [y for (x,y) in ordered_pairs]    
                               
df_user_cart_lists = \
    df_order_cart_counts.groupby('user_id').apply(cartSize_list).reset_index()
df_user_cart_lists.columns = ['user_id','cart_list']
                                 
df_user_cart_details = \
    df_order_cart_counts.groupby(['user_id'],as_index=False) \
                                 ['total_cart'].agg({'avg_cart': 'mean'}) \
                        .merge(df_user_cart_lists, how='left', on='user_id')

df_user_cart_details['last_cart'] = \
    df_user_cart_details['cart_list'].apply(lambda x: x[-1])
df_user_cart_details['delta_last_cart_avg'] = \
    df_user_cart_details['last_cart'] - df_user_cart_details['avg_cart'] 
    
#helper function - autocorrelation 
def auto_corr(series, lag):
    n = len(series)
    return pearsonr(series[0:n-lag], series[lag:n])[0]

df_user_cart_details['cart_autocorr_1'] = \
    df_user_cart_details['cart_list'].apply(auto_corr, args=[1])

#normalized interaction b/w cartsize autocorr and delta of last cart from avg
df_user_cart_details['cart_ac_deltalast_norm'] = \
    df_user_cart_details['cart_autocorr_1'] * \
    (df_user_cart_details['delta_last_cart_avg'] / df_user_cart_details['avg_cart'])

df_user_cart_details = df_user_cart_details.drop('cart_list',axis=1)
                                                                    
#Bring extracted cart data back into prior data and create a priority score:
#Higher priority score <-> added sooner to larger carts
#Then extract a df of average priority score by user-product                           
df_order_products_prior = pd.merge(df_order_products_prior, df_order_cart_counts, 
                                   how='left', 
                                   left_on=['order_number','user_id'], 
                                   right_on = ['order_number','user_id'])
 
df_order_products_prior['cart_priority'] = \
   1 - df_order_products_prior['add_to_cart_order'] / \
   (1 + df_order_products_prior['total_cart'])

#helper function to extract an ordered list of product priorities in each order
def tuple_prio_list(product_user_group):
    order_nums = product_user_group['order_number']
    priorities = product_user_group['cart_priority']
    ordered_pairs = sorted(zip(order_nums, priorities), key=lambda pair: pair[0])
    return ordered_pairs 

df_UP_prio_lists = \
    df_order_products_prior.groupby(['user_id','product_id']) \
                           .apply(tuple_prio_list).reset_index()
df_UP_prio_lists.columns = ['user_id','product_id','tuple_prio_list']
   
df_user_product_priorities = \
    df_order_products_prior.groupby(['product_id','user_id'],as_index=False) \
                                    ['cart_priority'].agg({'avg_priority': 'mean'})

tot_prior_orders = df_order_products_prior['order_id'].nunique()                                  
df_generic_product_details = \
    df_order_products_prior.groupby('product_id',as_index=False) \
                           .agg(OrderedDict([('cart_priority','mean'),
                                             ('order_id',(lambda x: x.nunique() / tot_prior_orders))])) \
                           .rename(columns={'cart_priority': 'avg_gen_priority', 
                                            'order_id': 'popularity'})
df_generic_product_freq = \
    df_prod_user_details.groupby('product_id',as_index=False) \
                                 ['order_freq'].agg({'avg_gen_freq': 'mean'})
                                   
'''Training Set Setup - pull extracted features into training df and use for
   further feature engineering
'''
                 
df_train_ord_prodList = \
    df_train.groupby(['order_id','user_id'],as_index=False) \
                     ['product_id'].agg({'product_list': (lambda x: set(x))})
df_train = df_train.drop(['product_id'],axis=1) 
df_train.drop_duplicates(subset='user_id',inplace=True)

df_train = pd.merge(df_train, df_train_ord_prodList, how='left', \
                    left_on = ['order_id','user_id'], right_on = ['order_id','user_id'])

df_train = pd.merge(df_train, df_prod_user_details, 
                    how='left', on='user_id')

df_train = pd.merge(df_train, df_user_product_priorities, how='left', 
                    left_on=['product_id','user_id'], right_on = ['product_id','user_id'])

df_train = pd.merge(df_train, df_UP_prio_lists, how='left', 
                    left_on=['product_id','user_id'], right_on = ['product_id','user_id'])
 
df_train = pd.merge(df_train, df_user_cart_details, 
                    how='left', on='user_id')

df_train = pd.merge(df_train, df_generic_product_details, 
                    how='left', on='product_id') 

df_train = pd.merge(df_train, df_generic_product_freq, 
                    how='left', on='product_id')

in_order = []
for product_id, product_list in zip(df_train['product_id'],df_train['product_list']):
    if product_id in product_list:
        in_order.append(1)
    else:
        in_order.append(0)
df_train['in_order'] = in_order

#helper functions for calculating binary autocorrs on product orders
#and stats on past reorders 
def bin_auto_corr(series, lag):
    n = len(series)
    return np.dot(series[0:n-lag], series[lag:n]) / (n-lag)

def reorder_stats(orders_list_bin):
    reorder_count, order_count = 0, 0
    
    for i in range(len(orders_list_bin)-1):
        if (orders_list_bin[i] == 1):
            order_count += 1
            if (orders_list_bin[i+1] == 1):
                reorder_count += 1
    
    if (orders_list_bin[-1] == 1 and order_count == 0):
        return 0, .05
    
    reorder_prob = reorder_count / order_count
            
    return reorder_count, reorder_prob
 
#more features to add
last4_freq, last6_freq, last10_freq = [], [], []
auto_corr_1, auto_corr_2, auto_corr_3 = [], [], []
past_reorders, reorder_prob, orders_since, streak = [], [], [], []
prev1_order, prev2_order, prev3_order = [], [], []
last_prio, last4_prio, last6_prio, last10_prio = [], [], [], []
 
#looping through ordered list of product priorities and total order count -
#prio list lets us determine if a product occured in an order and its priority                  
for tuple_prio_list, total_orders \
    in zip(df_train['tuple_prio_list'],df_train['total_orders']):
    
    bin_list, bin_list_prio = [-1] * total_orders, [0] * total_orders 
    for x, y in tuple_prio_list:
        bin_list[x-1] = 1
        bin_list_prio[x-1] = y
                
    last4_freq.append(bin_list[-4:].count(1)/4)
    last6_freq.append(bin_list[-6:].count(1)/6)
    last10_freq.append(bin_list[-10:].count(1)/10)
    
    auto_corr_1.append(bin_auto_corr(bin_list,1))
    auto_corr_2.append(bin_auto_corr(bin_list,2))
    auto_corr_3.append(bin_auto_corr(bin_list,3))
     
    reo_count, reo_prob = reorder_stats(bin_list)
    past_reorders.append(reo_count)
    reorder_prob.append(reo_prob)
    since = bin_list[::-1].index(1) 
    orders_since.append(since)
    
    last_prio.append(bin_list_prio[-1])
    last4_prio.append(np.mean(bin_list_prio[-4:]))
    last6_prio.append(np.mean(bin_list_prio[-6:]))
    last10_prio.append(np.mean(bin_list_prio[-10:]))
    
    try:
        streak_len = bin_list[::-1].index(-1)
    except: #occured in every order
        streak_len = len(bin_list)
    streak.append(streak_len)
 
    prev1, prev2, prev3 = (-1, -1, -1)
    if (since == 0):
        prev1 = 1
    elif (since == 1):
        prev2 = 1
    elif (since == 2):
        prev3 = 1
        
    prev1_order.append(prev1)
    prev2_order.append(prev2)
    prev3_order.append(prev3)

df_train['last4_freq'] = last4_freq
df_train['last6_freq'] = last6_freq
df_train['last10_freq'] = last10_freq

df_train['auto_corr_1'] = auto_corr_1
df_train['auto_corr_2'] = auto_corr_2
df_train['auto_corr_3'] = auto_corr_3
df_train['past_reorders'] = past_reorders
df_train['reorder_prob'] = reorder_prob
df_train['orders_since'] = orders_since
df_train['streak'] = streak
df_train['prev1_order'] = prev1_order 
df_train['prev2_order'] = prev2_order  
df_train['prev3_order'] = prev3_order
        
df_train['last_prio'] = last_prio
df_train['last4_prio'] = last4_prio
df_train['last6_prio'] = last6_prio
df_train['last10_prio'] = last10_prio
           
df_train['avg_days_prior_delta'] = \
        df_train['days_since_prior_order'] - df_train['avg_days_since_prior_order']
df_train['avg_dow_delta'] = df_train['order_dow'] - df_train['avg_dow']
df_train['avg_hod_delta'] = df_train['order_hour_of_day'] - df_train['avg_hod']

df_train = df_train.drop(['add_to_cart_order','reordered','order_number',
                          'times_ordered','orders_list','tuple_prio_list'],axis=1)
                      
df_train['auto_corr_prev1'] = df_train['auto_corr_1'] * df_train['prev1_order']
df_train['auto_corr_prev2'] = df_train['auto_corr_2'] * df_train['prev2_order']
df_train['auto_corr_prev3'] = df_train['auto_corr_3'] * df_train['prev3_order']

df_train.fillna(0,inplace=True)

#bring in product category info
df_train = pd.merge(df_train, df_products, on='product_id', how='left')
df_train.to_csv('df_train_full.csv',index=False)

'''Test Set Setup - repeat the df_train process but for the test set
'''

df_test = df_orders[df_orders['eval_set'] == 'test']
df_test = df_test[['order_id','user_id','order_number',
                     'order_dow','order_hour_of_day','days_since_prior_order']]
              
df_test = pd.merge(df_test, df_prod_user_details, 
                    how='left', on='user_id')

df_test = pd.merge(df_test, df_user_product_priorities, how='left', 
                    left_on=['product_id','user_id'], right_on = ['product_id','user_id'])

df_test = pd.merge(df_test, df_UP_prio_lists, how='left', 
                    left_on=['product_id','user_id'], right_on = ['product_id','user_id'])
 
df_test = pd.merge(df_test, df_user_cart_details, 
                    how='left', on='user_id')

df_test = pd.merge(df_test, df_generic_product_details, 
                    how='left', on='product_id') 

df_test = pd.merge(df_test, df_generic_product_freq, 
                    how='left', on='product_id')
                           
last4_freq, last6_freq, last10_freq = [], [], []
auto_corr_1, auto_corr_2, auto_corr_3 = [], [], []
past_reorders, reorder_prob, orders_since, streak = [], [], [], []
prev1_order, prev2_order, prev3_order = [], [], []
last_prio, last4_prio, last6_prio, last10_prio = [], [], [], []
                      
for tuple_prio_list, total_orders \
    in zip(df_test['tuple_prio_list'],df_test['total_orders']):
    
    bin_list, bin_list_prio = [-1] * total_orders, [0] * total_orders 
    for x, y in tuple_prio_list:
        bin_list[x-1] = 1
        bin_list_prio[x-1] = y
                
    last4_freq.append(bin_list[-4:].count(1)/4)
    last6_freq.append(bin_list[-6:].count(1)/6)
    last10_freq.append(bin_list[-10:].count(1)/10)
    
    auto_corr_1.append(bin_auto_corr(bin_list,1))
    auto_corr_2.append(bin_auto_corr(bin_list,2))
    auto_corr_3.append(bin_auto_corr(bin_list,3))
     
    reo_count, reo_prob = reorder_stats(bin_list)
    past_reorders.append(reo_count)
    reorder_prob.append(reo_prob)
    since = bin_list[::-1].index(1) 
    orders_since.append(since)
    
    last_prio.append(bin_list_prio[-1])
    last4_prio.append(np.mean(bin_list_prio[-4:]))
    last6_prio.append(np.mean(bin_list_prio[-6:]))
    last10_prio.append(np.mean(bin_list_prio[-10:]))
    
    try:
        streak_len = bin_list[::-1].index(-1)
    except:
        streak_len = len(bin_list)
    streak.append(streak_len)
 
    prev1, prev2, prev3 = (-1, -1, -1)
    if (since == 0):
        prev1 = 1
    elif (since == 1):
        prev2 = 1
    elif (since == 2):
        prev3 = 1
        
    prev1_order.append(prev1)
    prev2_order.append(prev2)
    prev3_order.append(prev3)

df_test['last4_freq'] = last4_freq
df_test['last6_freq'] = last6_freq
df_test['last10_freq'] = last10_freq

df_test['auto_corr_1'] = auto_corr_1
df_test['auto_corr_2'] = auto_corr_2
df_test['auto_corr_3'] = auto_corr_3
df_test['past_reorders'] = past_reorders
df_test['reorder_prob'] = reorder_prob
df_test['orders_since'] = orders_since
df_test['streak'] = streak
df_test['prev1_order'] = prev1_order 
df_test['prev2_order'] = prev2_order  
df_test['prev3_order'] = prev3_order
        
df_test['last_prio'] = last_prio
df_test['last4_prio'] = last4_prio
df_test['last6_prio'] = last6_prio
df_test['last10_prio'] = last10_prio
           
df_test['avg_days_prior_delta'] = \
        df_test['days_since_prior_order'] - df_test['avg_days_since_prior_order']
df_test['avg_dow_delta'] = df_test['order_dow'] - df_test['avg_dow']
df_test['avg_hod_delta'] = df_test['order_hour_of_day'] - df_test['avg_hod']

df_test = df_test.drop(['order_number','times_ordered',
                        'orders_list','tuple_prio_list'],axis=1)
          
df_test['auto_corr_prev1'] = df_test['auto_corr_1'] * df_test['prev1_order']
df_test['auto_corr_prev2'] = df_test['auto_corr_2'] * df_test['prev2_order']
df_test['auto_corr_prev3'] = df_test['auto_corr_3'] * df_test['prev3_order']

df_test.fillna(0,inplace=True)

df_test = pd.merge(df_test, df_products, on='product_id', how='left')
df_test.to_csv('df_test.csv',index=False)
