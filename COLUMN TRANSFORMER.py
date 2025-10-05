# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 14:02:35 2025

@author: richi
"""

import numpy as np
import pandas as pd


df = pd.read_csv(r"D:\MACHINE LEARNING DATASETS\covid_toy.csv")

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

df.isnull().sum()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df.drop(columns=['has_covid']),
                                                 df['has_covid'],
                                                test_size=0.2)


from sklearn.compose import ColumnTransformer
transformer = ColumnTransformer(transformers=[
    ('tnf1',SimpleImputer(),['fever']),
    ('tnf2',OrdinalEncoder(categories=[['Mild','Strong']]),['cough']),
    ('tnf3',OneHotEncoder(sparse_output=False,drop='first'),['gender','city'])
],remainder='passthrough')


transformer.fit_transform(X_train).shape

transformer.transform(X_test).shape



















