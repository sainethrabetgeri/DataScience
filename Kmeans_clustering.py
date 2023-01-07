#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 10:54:22 2023

@author: nethrachekuri
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv(r"C:\Users\neth\Desktop\Mall.csv")

x = dataset.iloc[:,[2,4]].values

from sklearn.cluster import KMeans

wcss = []

for i in range(1,11):
    kmeans = KMeans(n_cluster=i,init='k-means++',random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.titile('Elbow method')
plt.xlabel('Clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_cluster=5,init='k-means++',random_state=42)
y_kmeans=kmeans.fit_predict(x)

plt.scatter(x[y_kmeans == 0,0], x[y_kmeans == 0,1], s=100,c='red', label='Cluster 1')
plt.scatter(x[y_kmeans == 1,0], x[y_kmeans == 1,1], s=100,c='green', label='Cluster 2')
plt.scatter(x[y_kmeans == 2,0], x[y_kmeans == 2,1], s=100,c='blue', label='Cluster 3')
plt.scatter(x[y_kmeans == 3,0], x[y_kmeans == 3,1], s=100,c='yellow', label='Cluster 4')
plt.scatter(x[y_kmeans == 4,0], x[y_kmeans == 4,1], s=100,c='magenta', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='cyan')
plt.titile('Customers')
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()