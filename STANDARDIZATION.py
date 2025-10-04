# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 15:31:11 2025

@author: richi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\richi\OneDrive\Desktop\Social_Network_Ads (1).csv")

df = df.iloc[:,2:]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(
                                                  df.drop('Purchased',axis=1),
                                                  df['Purchased'],test_size=0.3,
                                                  random_state=0
                                                  )


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train_Scaled = scaler.transform(X_train)
X_test_Scaled = scaler.transform(X_test)

scaler.mean_


X_train_Scaled = pd.DataFrame(X_train_Scaled,columns=X_train.columns)
X_test_Scaled = pd.DataFrame(X_test_Scaled,columns=X_test.columns)

np.round(X_train.describe(),1)
np.round(X_train_Scaled.describe(),1)


figs, (ax1,ax2) = plt.subplots(ncols=2, figsize=(12,5))
ax1.scatter(X_train['Age'],X_train['EstimatedSalary'])
ax1.set_title("Before Scaling")
ax2.scatter(X_train_Scaled['Age'],X_train_Scaled['EstimatedSalary'],color='red')
ax2.set_title("After Scaling")
plt.show()

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# before scaling
ax1.set_title('Before Scaling')
sns.kdeplot(X_train['Age'], ax=ax1)
sns.kdeplot(X_train['EstimatedSalary'], ax=ax1)

# after scaling
ax2.set_title('After Standard Scaling')
sns.kdeplot(X_train_Scaled['Age'], ax=ax2)
sns.kdeplot(X_train_Scaled['EstimatedSalary'], ax=ax2)
plt.show()


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
# before scaling
ax1.set_title('Age Distribution Before Scaling')
sns.kdeplot(X_train['Age'], ax=ax1)
# after scaling
ax2.set_title('Age Distribution After Standard Scaling')
sns.kdeplot(X_train_Scaled['Age'], ax=ax2)
plt.show()

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
# before scaling
ax1.set_title('Salary Distribution Before Scaling')
sns.kdeplot(X_train['EstimatedSalary'], ax=ax1)
# after scaling
ax2.set_title('Salary Distribution Standard Scaling')
sns.kdeplot(X_train_Scaled['EstimatedSalary'], ax=ax2)
plt.show()

