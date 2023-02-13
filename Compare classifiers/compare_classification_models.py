
"""
Created on Sun Jan  1 20:53:48 2023

@author: tle19
"""
import os
os.chdir('/Users/tiarnalee/Documents/cosm/')
from sklearn.exceptions import ConvergenceWarning
from warnings import simplefilter
import dim_reduction as cl
import sys
import datetime
from time import time
import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np
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
# Create array of stacked images
# X=np.mean(stacked_images, axis=3).reshape(len(stacked_images), 600**2)
#
# %%
simplefilter("ignore", category=ConvergenceWarning)

# List of classification models and dimensionality reduction techniques
classification = ('RFC', 'LSVC', 'MLP', 'Logistic regression', 'LDA', 'Adaboost')
dimred = ('PCA', 'LDA', 'Kernel PCA')
n_class = len(classification)
n_dimred = len(dimred)
repeats=10

tic = time()
#array for standard error (error between runs)
yerr = np.zeros((n_dimred, n_class))
results = np.zeros((n_dimred, n_class))
acc = np.zeros((repeats, n_class))

# Loops through dimensionality reduction and classification methods
# NB: for each of the dimensionality methods, the principle components should be set as an integer, not a proportion of variance
# PCA followed by classification
_, _, _, X_pca = cl.comp_PCA(2, blue, red, green)
for method in np.arange(n_class):
    for repeat in np.arange(repeats):
        print('\nPCA Epoch {}/{}, Method {}/{}'.format(repeat+1, repeats, method+1, n_class))
        acc[repeat, method] = cl.classify(X_pca, num_labels, method)
    # Find std and mean of each method
    yerr[0, method] = acc[:, method].std(axis=0)
    results[0, method] = acc[:, method].mean(0)

# LDA followed by classification
_, _, _, X_lda = cl.comp_LDA(5, blue, red, green, num_labels)
for method in np.arange(n_class):
    for repeat in np.arange(repeats):
        print('\nLDA Epoch {}/{}, Method {}/{}'.format(repeat+1, repeats, method+1, n_class))
        acc[repeat, method] = cl.classify(X_lda, num_labels, method)
    # Find std and mean of each method
    yerr[1, method] = acc[:, method].std(axis=0)
    results[1, method] = acc[:, method].mean(0)

# Kernel PCA followed by classification
_, _, _, X_kern = cl.comp_kPCA(2, blue, red, green)
for method in np.arange(n_class):
    for repeat in np.arange(repeats):
        print('\nKernel PCA {}/{}, Method {}/{}'.format(repeat+1, repeats, method+1, n_class))
        acc[repeat, method] = cl.classify(X_kern, num_labels, method)
    # Find std and mean of each method
    yerr[2, method] = acc[:, method].std(axis=0)
    results[2, method] = acc[:, method].mean(0)

# calculate standard error from error 
yerr = (yerr*1.96)/np.sqrt(repeats) #1.96 is 2 standard deviations

toc = time()
sec = toc-tic
res = datetime.timedelta(seconds=sec)
print(f"Done in {res}s")

best = np.amax(results)
print('Highest accuracy is {} from {} classification on {}'.format(round(best, 5),
      classification[(np.where(results == best)[1][0])], dimred[(np.where(results == best)[0][0])]))

print('STD = {}'.format(round(yerr[np.where(results == best)[
      0][0], (np.where(results == best)[1][0])], 5)))

# plot results
fig = plt.figure()
width = 0.25
rects2 = plt.bar(np.arange(n_class), results[0, :], yerr=yerr[0, :],
                 width=width, label='PCA', alpha=0.75, ecolor='black', capsize=3)
rects3 = plt.bar(np.arange(n_class) + width,
                 results[1, :], yerr=yerr[1, :], width=width, label='LDA', alpha=0.6, ecolor='black', capsize=3)
rects4 = plt.bar(np.arange(n_class) + 2*width, results[2, :], yerr=yerr[2, :], width=width,
                 label='Kernel PCA', alpha=0.6, ecolor='black', capsize=3, hatch="/", edgecolor='black')
plt.xticks(np.arange(n_class)+width, classification)
plt.xlabel('Classification method')
plt.ylabel('Accuracy')
fig.suptitle(
    'Classification model performance on hair type (n= {})'.format(len(im_paths)))
# fig.suptitle('Classification on male season test set ({} runs)'.format(repeats))
plt.tight_layout()
plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
           mode="expand", borderaxespad=0, ncol=n_dimred)


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.annotate('{}'.format(round(height, 2)),
                     xy=(rect.get_x() + rect.get_width() / 2, height*0.5),
                     xytext=(0, 0),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')


autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
plt.show()
