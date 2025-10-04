# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 10:20:28 2025

@author: richi
"""

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\richi\OneDrive\Desktop\wine_data.csv",header=None,usecols=[0,1,2])
df.columns = ['Class label', 'Alcohol', 'Malic acid']

sns.kdeplot(df['Alcohol'])
sns.kdeplot(df['Malic acid'])

color_dict={1:'red',3:'green',2:'blue'}
sns.scatterplot(
    data=df,
    x='Alcohol',
    y='Malic acid',
    hue='Class label',
    palette=color_dict
)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df.drop('Class label',axis=1),
                                                 df['Class label'],
                                                 test_size = 0.3,
                                                 random_state=0)

X_train.shape,X_test.shape

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

np.round(X_train.describe(), 1)

np.round(X_train_scaled.describe(),1)



fig , (ax1,ax2) = plt.subplots(ncols=2,figsize = (12,5))
ax1.scatter(X_train['Alcohol'],X_train['Malic acid'],c=y_train)
ax1.set_title('before scaling')
ax2.scatter(X_train_scaled['Alcohol'],X_train_scaled['Malic acid'],c=y_train)
ax2.set_title('after scaling')


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# before scaling
ax1.set_title('Before Scaling')
sns.kdeplot(X_train['Alcohol'], ax=ax1)
sns.kdeplot(X_train['Malic acid'], ax=ax1)

# after scaling
ax2.set_title('After Standard Scaling')
sns.kdeplot(X_train_scaled['Alcohol'], ax=ax2)
sns.kdeplot(X_train_scaled['Malic acid'], ax=ax2)
plt.show()




























