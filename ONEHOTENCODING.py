# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 13:57:32 2025

@author: richi
"""

import numpy as np
import pandas as pd


df = pd.read_csv(r"D:\MACHINE LEARNING DATASETS\cars.csv")

df['owner'].value_counts()

#1. OneHotEncoding using Pandas
df_encoded = pd.get_dummies(df, columns=['fuel', 'owner'])
# Convert only boolean columns to int
bool_cols = df_encoded.select_dtypes(bool).columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)


#2. K-1 OneHotEncoding
df_encoded_1 = pd.get_dummies(df, columns=['fuel', 'owner'],drop_first=True)
# Convert only boolean columns to int
bool_cols_1 = df_encoded_1.select_dtypes(bool).columns
df_encoded_1[bool_cols_1] = df_encoded_1[bool_cols_1].astype(int)



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df.iloc[:,0:4],df.iloc[:,-1],test_size=0.2,random_state=2)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(drop='first', sparse_output=False, dtype=np.int32)

X_train_new = ohe.fit_transform(X_train[['fuel','owner']])
X_test_new = ohe.transform(X_test[['fuel','owner']])
df_1 = np.hstack((X_train[['brand','km_driven']].values,X_train_new))



