# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 12:40:04 2025

@author: richi
"""

import numpy as np
import pandas as pd

df = pd.read_csv(r"C:\Users\richi\OneDrive\Desktop\customer.csv")

df = df.iloc[:,2:]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df.iloc[:,0:2],
                                                 df.iloc[:,-1],
                                                 test_size = 0.3,
                                                 random_state=0)

#ORDINAL ENCODER

from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder(categories=[['Poor','Average','Good'],
                                ['School','UG','PG']])
oe.fit(X_train)
X_train_oe = oe.transform(X_train)
X_test_oe = oe.transform(X_test)

#LABEL ENCODER

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y_train)
y_train_le = le.transform(y_train)
y_test_le = le.transform(y_test)


















