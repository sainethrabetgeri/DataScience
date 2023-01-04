#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 14:57:00 2022

@author: nethrachekuri
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv(r"C:\Users\neth\Desktop\LR.csv")

x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:-1].values
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(penalty='12',solve='sag')
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)
bias=classifier.score(x_train,y_train)
bias
variance= classifier.score(x_test,y_test)
variance

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

cm = confusion_matrix(y_test,y_pred)
cr = classification_report(y_test, y_pred)

from sklearn.metrics import accuracy_score
c = accuracy_score(y_test, y_pred)


#Future Prediction
dataset1 = pd.read_csv(r"C:\Users\neth\Desktop\FLR.csv")
d2 = dataset1.copy()
dataset1 = dataset1.iloc[:,[2:3]].values

sc = StandardScaler()
M = sc.fit_transform(dataset1)

d2['y_pred'] = classifier.predict(M)
d2.to_csv('Final.csv')