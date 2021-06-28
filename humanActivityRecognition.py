# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 15:38:56 2017

@author: Daniel.McGlynn
Splits data into training and test sets.
Performs Principal component analysis
Calcs MSE from 10-fold cross validation for all principal components
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from scipy.signal import wiener
import numpy.polynomial.polynomial as poly
from sklearn import linear_model
import sklearn.metrics
import collections
from random import randint
import csv
from sklearn.model_selection import KFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report

data = pd.read_excel('X_train.xlsx', header=0, sheetname='training_set', parse_cols="A:JF")


test_data = pd.read_excel('X_test.xlsx', header=0, sheetname='test_set', parse_cols="A:JF")

labels_data = pd.read_excel('activity_labels.xlsx', header=0, sheetname='activity_labels', usecols=["activity label"])
#df = data.dropna()
#df_test = data.dropna()

ar = data.values
ar_test = test_data.values

normalisedData = preprocessing.StandardScaler().fit_transform(ar)
normalisedData_test = preprocessing.StandardScaler().fit_transform(ar_test)

outlier_rows, outlier_columns = np.where(np.abs(normalisedData)>3)
outlier_rows_test, outlier_columns_test = np.where(np.abs(normalisedData_test)>3)

#print(list(zip(outlier_rows, outlier_columns)))

#Remove rows containing outliers
data = data.drop(data.index[outlier_rows])
test_data = test_data.drop(test_data.index[outlier_rows_test])

ar = data.values
ar_test = test_data.values

normalisedData = preprocessing.StandardScaler().fit_transform(ar)
normalisedData_test = preprocessing.StandardScaler().fit_transform(ar_test)
#drop rows containing outliers
#df = df.drop(df.index[outlier_rows])


plt.figure(figsize=(50,38), num=1)

X = data.iloc[:,1:265]
y_train = data["Activity"]

X_test = test_data.iloc[:,1:265]
y_test = test_data["Activity"]

#Scale dataset
ar = X.values
normalisedData = preprocessing.StandardScaler().fit_transform(ar)

#Principal component analysis
pca1 = PCA()
x_pca_p = pca1.fit_transform(normalisedData)
x_pca_p_test = pca1.fit_transform(normalisedData_test)

#random forest classifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
model_forest = clf.fit(x_pca_p[:,:70], y_train)
predictions_forest = clf.predict(x_pca_p_test[:,:70])

#confusion matrix random forest
fig = plt.figure(figsize=(25,19), num=1)
ax = plt.gca()
cm_forest = metrics.confusion_matrix(test_data["Activity"], predictions_forest)
        
img = ax.matshow(cm_forest, cmap=plt.cm.autumn)
fig.colorbar(img, fraction=0.045)
for x in range(cm_forest.shape[0]):
    for y in range(cm_forest.shape[1]):
        plt.text(y,x, "%0.2f" % cm_forest[x,y], size=22, color='black', ha="center", va="center")
    
plt.yticks(np.arange(0,5.5),range(0,5))
plt.xticks(np.arange(0,5.5),range(0,5))
ax.set_xticklabels(labels_data['activity label'].values,rotation=90, fontsize=22)
ax.set_yticklabels(labels_data['activity label'].values, fontsize=22)
ax.set_xlabel('Predicted', fontsize=22)
ax.set_ylabel('Actual', fontsize=22)
fig.savefig('confustionMatrixRandomForest.png', bbox_inches='tight')


#K nearest neighbour classifier
neigh = KNeighborsClassifier(n_neighbors=3)
model_neigh = neigh.fit(x_pca_p[:,:70], y_train) 
predictions_neigh = neigh.predict(x_pca_p_test[:,:70])

#confusion matrix Nearest Negihbours
fig = plt.figure(figsize=(25,19), num=1)
ax = plt.gca()
#ax = fig.add_subplot(2,2,2)
cm_neigh = metrics.confusion_matrix(test_data["Activity"], predictions_neigh)
        
img = ax.matshow(cm_neigh, cmap=plt.cm.autumn)
fig.colorbar(img, fraction=0.045)
for x in range(cm_neigh.shape[0]):
    for y in range(cm_neigh.shape[1]):
        plt.text(y,x, "%0.2f" % cm_neigh[x,y], size=22, color='black', ha="center", va="center")

plt.yticks(np.arange(0,5.5),range(0,5))
plt.xticks(np.arange(0,5.5),range(0,5))
ax.set_xticklabels(labels_data['activity label'].values,rotation=90, fontsize=22)
ax.set_yticklabels(labels_data['activity label'].values, fontsize=22)
ax.set_xlabel('Predicted', fontsize=22)
ax.set_ylabel('Actual', fontsize=22)
fig.savefig('confustionMatrixNearestNeighbours.png', bbox_inches='tight')


#support vector machine classifier
clf = svm.SVC()
model = clf.fit(x_pca_p[:,:70], y_train) 
predictions = clf.predict(x_pca_p_test[:,:70]) 

#confusion matrix SVM
fig = plt.figure(figsize=(25,19), num=1)
ax = plt.gca()
#ax = fig.add_subplot(2,2,3)
cm = metrics.confusion_matrix(test_data["Activity"], predictions)

img = ax.matshow(cm, cmap=plt.cm.autumn)
fig.colorbar(img, fraction=0.045)
for x in range(cm.shape[0]):
    for y in range(cm.shape[1]):
        plt.text(y,x, "%0.2f" % cm[x,y], size=22, color='black', ha="center", va="center")

plt.yticks(np.arange(0,5.5),range(0,5))
plt.xticks(np.arange(0,5.5),range(0,5))
ax.set_xticklabels(labels_data['activity label'].values,rotation=90, fontsize=22)
ax.set_yticklabels(labels_data['activity label'].values, fontsize=22)
ax.set_xlabel('Predicted', fontsize=22)
ax.set_ylabel('Actual', fontsize=22)
fig.savefig('confustionMatrixSVM.png', bbox_inches='tight')

#plot individual and cumulative explained variance 
fig = plt.figure(figsize=(25,19), num=1)
ax1 = fig.add_subplot(2,2,4)
ax1 = plt.gca()
plt.grid()
eig_vals = pca1.explained_variance_ratio_*100 #eigen values
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
plt.bar(range(1,267), var_exp, alpha=0.5, align='center',label='individual explained variance')

labels = ['PC-1','PC-2','PC-3','PC-4']
#plt.xticks(np.arange(0,265.5),range(0,265))
#ax.set_xticklabels(labels,rotation='vertical')
ax1.legend(loc='upper left')
plt.xlim(0,75)
ax1.set_xlabel('Principal components')
ax1.set_ylabel('Individual variance')

ax2 = ax1.twinx()
plt.grid()
plt.plot(range(1,267), np.cumsum(eig_vals), 'o-', label='cumulative explained variance')
ax2.legend(loc='upper right')
plt.xlim(0,75)
ax2.set_ylabel('Cumulative variance')
plt.show()
fig.savefig('PCA.png', bbox_inches='tight')

#accuracy, presicion, recall and f1 score SVM
print("Accuracy:", metrics.accuracy_score(test_data["Activity"], predictions))
print("Precision:", metrics.precision_score(test_data["Activity"], predictions, average='weighted'))
print("Recall:", metrics.recall_score(test_data["Activity"], predictions, average='weighted'))
print("F1 score:", metrics.f1_score(test_data["Activity"], predictions, average='weighted'))

print(classification_report(test_data["Activity"], predictions, target_names=labels_data['activity label'].values))

#accuracy, presicion, recall and f1 score Nearest Neighbours
print("Accuracy:", metrics.accuracy_score(test_data["Activity"], predictions_neigh))
print("Precision:", metrics.precision_score(test_data["Activity"], predictions_neigh, average='weighted'))
print("Recall:", metrics.recall_score(test_data["Activity"], predictions_neigh, average='weighted'))
print("F1 score:", metrics.f1_score(test_data["Activity"], predictions_neigh, average='weighted'))

print(classification_report(test_data["Activity"], predictions_neigh, target_names=labels_data['activity label'].values))

#accuracy, presicion, recall and f1 score Random Forest
print("Accuracy:", metrics.accuracy_score(test_data["Activity"], predictions_forest))
print("Precision:", metrics.precision_score(test_data["Activity"], predictions_forest, average='weighted'))
print("Recall:", metrics.recall_score(test_data["Activity"], predictions_forest, average='weighted'))
print("F1 score:", metrics.f1_score(test_data["Activity"], predictions_forest, average='weighted'))

print(classification_report(test_data["Activity"], predictions_forest, target_names=labels_data['activity label'].values))

#Mean absolute error SVM
MAE = sklearn.metrics.mean_absolute_error(y_test, predictions, multioutput='uniform_average')
print('Mean Absolute Error SVM:', "%0.2f" % MAE)

#Mean absolute error nearest neighbours
MAE = sklearn.metrics.mean_absolute_error(y_test, predictions_neigh, multioutput='uniform_average')
print('Mean Absolute Error (Nearest neighbours):', "%0.2f" % MAE)

#Mean absolute error random forest
MAE = sklearn.metrics.mean_absolute_error(y_test, predictions_forest, multioutput='uniform_average')
print('Mean Absolute Error (Random forest):', "%0.2f" % MAE)
print(' ')

#Cross validation SVM
score = cross_val_score(model, X, y_train, cv=10, scoring="neg_mean_squared_error")
print("Mean Squared Error SVM: %0.2f" % (-score.mean()))

#Cross validation nearest neighbours
score = cross_val_score(model_neigh, X, y_train, cv=10, scoring="neg_mean_squared_error")
print("Mean Squared Error (Nearest neighbours): %0.2f" % (-score.mean()))

#Cross validation random forest
score = cross_val_score(model_forest, X, y_train, cv=10, scoring="neg_mean_squared_error")
print("Mean Squared Error (Random Forest): %0.2f" % (-score.mean()))
print(' ')

#k-fold cross validation SVM
kf = KFold(n_splits = 10, random_state = 1)
scores = cross_val_score(model, x_pca_p[:,:70], y_train, scoring='neg_mean_squared_error', cv=kf)
print('SVM MSE\t\t\t%10.3f' %(-scores.mean()))

#k-fold cross validation Nearest Neighbours
kf = KFold(n_splits = 10, random_state = 1)
scores = cross_val_score(model_neigh, x_pca_p[:,:70], y_train, scoring='neg_mean_squared_error', cv=kf)
print('Nearest Neighours MSE:\t\t%5.3f' %(-scores.mean()))

#k-fold cross validation random forest
kf = KFold(n_splits = 10, random_state = 1)
scores = cross_val_score(model_forest, x_pca_p[:,:70], y_train, scoring='neg_mean_squared_error', cv=kf)
print('Random Forest MSE:\t\t%5.3f' %(-scores.mean()))
print(' ')

fig.savefig('humanActivityClassificationConfusionMatrix.png', bbox_inches='tight')