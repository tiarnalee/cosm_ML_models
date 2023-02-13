#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 19:18:22 2023

@author: tiarnalee
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 20:53:48 2023

@author: tle19
"""
import os
os.chdir('/Users/tiarnalee/Documents/cosm/')
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
from warnings import simplefilter
import dim_reduction as cl
import sys
import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import pickle
import random

classes = ['3A', '3B', '3C', '4A', '4B', '4C']

training_validation_path = '/Users/tiarnalee/Documents/cosm/segmented_images/'

im_paths, labels, num_labels = [], [], []

# Find image paths and labels
print('Reading data...')
# Get path for every image
_ = [im_paths.append(glob.glob(os.path.join(training_validation_path, i, "*"))) for i in classes]
im_paths = np.hstack(im_paths)
# Get labels from image paths
_ = [labels.append(j.split('/')[-2]) for j in im_paths]
labels = np.hstack(labels)
#Convert alphabetic labels to numbers
_ = [num_labels.append(np.where(i == np.array(classes))[0][0]) for i in labels]
num_labels = np.hstack(num_labels)

print('Separating data into R/G/B...')

# Read RGB channels for each image (image sizes are 600x600)
blue, green, red = np.zeros((len(im_paths), 600*600)), np.zeros(
    (len(im_paths), 600*600)), np.zeros((len(im_paths), 600*600))

# The images can be read and stacked instead of separating into RGB channels.
# Uncommment commented code in this section
# stacked_images = np.zeros((len(im_paths), 600, 600, 3))
for i, path in enumerate(im_paths):
    img=cv2.imread(path)
    
    # add blurring and Gaussian noise to the images
    if np.random.uniform(0,1, 1)<0.1:
        img = cv2.GaussianBlur(img, (random.randrange(1,10,2), random.randrange(1,10,2)), 0)
    elif np.random.uniform(0,1, 1)<0.1:
        img=cv2.blur(img, (random.randrange(1,10,2), random.randrange(1,10,2)))
 
 # Split the images into R/G/B channels
    b, g, r = cv2.split(img)

    blue[i, :] = b.flatten()/255
    green[i, :] = g.flatten()/255
    red[i, :] = r.flatten()/255

    # stacked_images[i, :, :, :] = cv2.imread(path)

del path, i, b, g, r

#%%
# Investigate LDA models
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
         
print('Reducing dimensions')

_,_,_, X_lda = cl.comp_LDA(5, blue, red, green, num_labels)

# To save the LDA fit, LDA must be performed without transforming the data 
# lda = LDA(n_components=5, solver='svd')
# red_fit = lda.fit(red, num_labels) 
# green_fit = lda.fit(green, num_labels)
# blue_fit = lda.fit(blue, num_labels)

# with open("LDA_model.pkl", "wb") as f:
#     for model in [red_fit, green_fit, blue_fit]:
#           pickle.dump(model, f)

#%%

print('Training...')
# The model can be trained multiple times and the results averaged
cm=[] #Confuson matrix

for i in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(
        X_lda, num_labels, test_size=0.2, train_size=0.8, stratify=num_labels)
    
    # Model has to be re-initialised in every iteration 
    # clf = RandomForestClassifier(n_estimators=1000, max_depth=5, min_samples_leaf=2)
    clf = KNeighborsClassifier(n_neighbors=3, metric='l2')
    # clf=RadiusNeighborsClassifier(radius=0.5)
    
    # The data can be rescaled to mean and standard deviation
    scaler = StandardScaler()
    X_train=scaler.fit_transform(X_train)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # Results are appended to confusion matrix
    cm.append(confusion_matrix(y_test, y_pred, labels=clf.classes_, normalize='true' ))

# Plot the mean results
plt.figure()
disp=ConfusionMatrixDisplay(np.mean(np.dstack(cm), axis=2), display_labels=classes).plot()

# Save the classifier
# with open('classifier.pkl','wb') as f:
    # pickle.dump(clf,f)


#%% Use t-SNE to visualise clusters (Clustering is unstable and inconsistent)

# We want to get TSNE embedding with 2 dimensions
# n_components = 2
# tsne = TSNE(n_components, init='random', learning_rate='auto', method='exact', n_iter_without_progress=100, n_iter=2000)
# tsne_result = tsne.fit_transform(X_lda)

# # Plot the result of our TSNE with the label color coded
# tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': num_labels})
# fig, ax = plt.subplots(1)
# sns.scatterplot(x='tsne_1', y='tsne_2', palette=plt.cm.get_cmap('tab20', 6),hue='label', data=tsne_result_df, ax=ax,s=120, alpha=0.8)
# lim = (tsne_result.min()-5, tsne_result.max()+5)
# ax.set_xlim(lim)
# ax.set_ylim(lim)
# ax.set_aspect('equal')
# ax.legend(classes, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
