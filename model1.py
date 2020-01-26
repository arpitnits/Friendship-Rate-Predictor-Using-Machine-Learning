# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 16:36:46 2020

@author: Arpit
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_excel('G:/Projects\Machine Learning/Lazarus/Project_Data.xlsx')
#check=dataset.iloc[:, 13].values
x=dataset.iloc[:, 1:14].values
print(x)

#Getting rid of categorical data of region...
category_Hot_Encoded=pd.get_dummies(x[:, 4])
x=np.append(x, category_Hot_Encoded, axis=1)
print(x)
x=np.delete(x, 4, 1)# This removes the region coulumm which in turn would be replaced by category_Hot_Encoded
#x=pd.DataFrame(x) to view it as  a data frame
#Getting rid of categorical data of Gender...
category_Hot_Encoded=pd.get_dummies(x[:, 1])
x=np.append(x, category_Hot_Encoded, axis=1)
print(x)
x=np.delete(x, 1, 1)# This removes the region coulumm which in turn would be replaced by category_Hot_Encoded
#x=pd.DataFrame(x) to view it as  a data frame
#Now we remove the female categorical data to avoid the categorical data trap
x=np.delete(x, 15, 1)
#x=pd.DataFrame(x)...just to visualize#Getting rid of categorical data of Gender...
#Getting rid of categorical data of Goals...
category_Hot_Encoded=pd.get_dummies(x[:, 4])
x=np.append(x, category_Hot_Encoded, axis=1)
print(x[1,: ])
x=np.delete(x, 4, 1)# This removes the region coulumm which in turn would be replaced by category_Hot_Encoded
#Getting rid of categorical data of Seats...
category_Hot_Encoded=pd.get_dummies(x[:, 7])
x=np.append(x, category_Hot_Encoded, axis=1)
print(x)
x=np.delete(x, 7, 1)# This removes the region coulumm which in turn would be replaced by category_Hot_Encoded


#Separating the hobbies separate dy comma
x_copy=x#This is used to maintain a x copy in object form
x=pd.DataFrame(x)

#Hobbies
clist = ['Game','Musical','Gym','Others']
for i in range(0, 325):
    list1 = x[3][i].split(', ')
    list2 = [0, 0, 0, 0]
    
    for j in range(0, 4):
        if clist[j] in list1:
            list2[j] = 1 
    x[3][i] = list2
    
    
tags = x[3].apply(pd.Series)
tags = tags.rename(columns = lambda a : 'tag_' + str(a))

x = pd.concat([x[:], tags[:]], axis=1)
x=pd.DataFrame(x)
x.rename(columns = {'tag_0':'Game'}, inplace = True)
x.rename(columns = {'tag_1':'Musical'}, inplace = True)
x.rename(columns = {'tag_2':'Gym'}, inplace = True)
x.rename(columns = {'tag_3':'Others'}, inplace = True)

#Delete column number 3(Hobbies) from the x...
x=x.drop(labels=3, axis=1)# This removes the region coulumm which in turn would be replaced by category_Hot_Encoded
x=x.drop(labels=5, axis=1)# This removes the friends coulumm

#Music taste
clist = ['M1','M2','M3','M4','M5','M6','M7','M8','Others']
for i in range(0, 325):
    list1 = x[8][i].split(', ')
    list2 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for j in range(0, 9):
        if clist[j] in list1:
            list2[j] = 1
            
    x[8][i] = list2
    
    
tags = x[8].apply(pd.Series)
tags = tags.rename(columns = lambda a : 'tag_' + str(a))

x = pd.concat([x[:], tags[:]], axis=1)
x=pd.DataFrame(x)

x.rename(columns = {'tag_0':'M1'}, inplace = True)
x.rename(columns = {'tag_1':'M2'}, inplace = True)
x.rename(columns = {'tag_2':'M3'}, inplace = True)
x.rename(columns = {'tag_3':'M4'}, inplace = True)
x.rename(columns = {'tag_4':'M5'}, inplace = True)
x.rename(columns = {'tag_5':'M6'}, inplace = True)
x.rename(columns = {'tag_6':'M7'}, inplace = True)
x.rename(columns = {'tag_7':'M8'}, inplace = True)
x.rename(columns = {'tag_8':'Others'}, inplace = True)
x=x.drop(labels=8, axis=1)# This removes the Music coulumm

#This drop is only temporary and needs to be removed as and when nasha is implemented...
x=x.drop(labels=6, axis=1)# This removes the Nasha coulumm
x=x.drop(labels=2, axis=1)# This removes the Room coulumm



'''Feature Scaling"'''

##Scholar ID
y=dataset.iloc[:, 1:2].values
#type(y[1][1])
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
y = sc_X.fit_transform(y)
x=np.append(x, y, axis=1)
x=pd.DataFrame(x)
x=x.drop(labels=0, axis=1)


##Expenditure
y=dataset.iloc[:, 3:4].values
#type(y[1][1])
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
y = sc_X.fit_transform(y)
x=np.append(x, y, axis=1)
x=pd.DataFrame(x)
x=x.drop(labels=0, axis=1)

'''Scaling Done'''

#Model
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(x)

'''Dimensionality Reduction'''
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
x = pca.fit_transform(x)
explained_variance = pca.explained_variance_ratio_


'''Visual'''
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of Students')
plt.xlabel('First feature)')
plt.ylabel('Second Feature')
plt.legend()
plt.show()