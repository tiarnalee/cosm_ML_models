# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 22:49:51 2023

@author: tle19
"""

import numpy as np
from sklearn.decomposition import PCA,  KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#Pre-processing
from sklearn.model_selection import train_test_split

#Classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

#Scoring

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import recall_score, confusion_matrix, accuracy_score, mean_squared_error

def comp_PCA(n_components, blue, red, green):
#initialize PCA with first n principal components
    pca = PCA(n_components)
     
    red_transformed = PCA(n_components).fit_transform(red) 
    green_transformed = PCA(n_components).fit_transform(green) 
    blue_transformed = PCA(n_components).fit_transform(blue)
    
    # Combine the values from the 3 channels
    coords=np.mean(np.dstack((red_transformed, green_transformed, blue_transformed)), axis=2)

    return red_transformed, green_transformed, blue_transformed, coords


def comp_LDA(n_components, blue, red, green, y):
#initialize LDA with first n principal components
    lda = LDA(n_components=n_components, solver='svd')
     
    red_transformed = lda.fit_transform(red, y) 
    green_transformed = lda.fit_transform(green, y)
    blue_transformed = lda.fit_transform(blue, y)
    # Combine the values from the 3 channels

    coords=np.mean(np.dstack((red_transformed, green_transformed, blue_transformed)), axis=2)
    
    return red_transformed, green_transformed, blue_transformed, coords

def comp_kPCA(n_components, blue, red, green):
#initialize kPCA with first n principal components
    kPCA = KernelPCA(n_components=n_components, kernel="linear", gamma=0.2,  alpha=0.3)
     
    red_transformed = kPCA.fit_transform(red) 
    green_transformed = kPCA.fit_transform(green)
    blue_transformed = kPCA.fit_transform(blue)
    # Combine the values from the 3 channels

    coords=np.mean(np.dstack((red_transformed, green_transformed, blue_transformed)), axis=2)

    return red_transformed, green_transformed, blue_transformed, coords
    
def RFC(X, y):
    # print('RFC:')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, stratify=y)
    
    # The depth and estimators are set at an arbitrary value as no grid search to find optimal parameters is performed
    clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, max_depth=5, min_samples_leaf=2)
    # The classifier is fit to the original data
    clf.fit(X_train, y_train)
# The (10-fold) cross-validated score is found
    scores = cross_val_score(clf, X_train, y_train, cv=10, scoring = 'accuracy')
    # print('Train accuracy:', round(scores.mean(),2))
    # Prediction and performance on test set
    y_pred=clf.predict(X_test)
    test_acc=accuracy_score(y_pred, y_test)
    # print('Test accuracy:', round(test_acc.mean(),2))
    print('Train/test accuracy %.3f/%.3f'% (round(scores.mean(),2), round(test_acc.mean(),2)))

    return test_acc
    
def LSVC(X, y):
    print('LSVC:')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, stratify=y)
    
    #Linear Support Vector classifier is fit to the data
    LSVC = LinearSVC(max_iter=5e4, verbose=0)
    LSVC.fit(X_train,y_train)
    # The (10-fold) cross-validated score is found

    scores = cross_val_score(LSVC, X_train, y_train, cv=10, scoring = 'accuracy')
     # Prediction and performance on test set

    y_pred=LSVC.predict(X_test) 
    test_acc=accuracy_score(y_pred, y_test)
    print('Train/test accuracy %.3f/%.3f'% (round(scores.mean(),2), round(test_acc.mean(),2)))

    return test_acc

def MLP_calc(X, y):
    print('MLP:')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, stratify=y)
    
    mlp = MLPClassifier(hidden_layer_sizes=(9,7), max_iter=1000, alpha=1e-4,
                solver='adam', verbose=0, tol=1e-5, 
                learning_rate_init=0.001, activation='tanh', learning_rate='adaptive')
    N_TRAIN_SAMPLES = X_train.shape[0]
    N_EPOCHS = 2000
    N_BATCH = 32
    N_CLASSES = np.unique(y_train)
    
    scores_train = []
    scores_test = []
    
    # EPOCH
    epoch = 0
    while epoch < N_EPOCHS:
        # print('epoch: ', epoch)
        # SHUFFLING
        random_perm = np.random.permutation(X_train.shape[0])
        mini_batch_index = 0
        while True:
            # MINI-BATCH
            indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
            mlp.partial_fit(X_train[indices], y_train[indices], classes=N_CLASSES)
            mini_batch_index += N_BATCH
    
            if mini_batch_index >= N_TRAIN_SAMPLES:
                break
    
        # SCORE TRAIN
        scores_train.append(mlp.score(X_train, y_train))
    
        # SCORE TEST
        scores_test.append(mlp.score(X_test, y_test))
    
        epoch += 1
    
    print('Training Accuracy:', round(mlp.score(X_train, y_train).mean(),2)) 
    # Prediction and performance on test set

    y_pred=mlp.predict(X_test)
    test_acc=accuracy_score(y_pred, y_test)
    print('Test Accuracy:', round(test_acc.mean(),2))
    return test_acc

def logis(X,y):
    print('Logistic Regression:')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, stratify=y)
    clf = LogisticRegression(class_weight='balanced', n_jobs=-1, C=5, max_iter=100)
    
    clf.fit(X_train, y_train)
    # C
    scores = cross_val_score(clf, X_train, y_train, cv=10, scoring = 'accuracy')
    y_pred=clf.predict(X_test)
    test_acc=accuracy_score(y_pred, y_test)
    print('Train/test accuracy %.3f/%.3f'% (round(scores.mean(),2), round(test_acc.mean(),2)))
        
    return test_acc

def LDA_class(X,y):
    print('LDA:')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, stratify=y)
    
    lda = LDA(n_components=1)
    X_lda= lda.fit(X_train, y_train)
    # The (10-fold) cross-validated score is found

    scores = cross_val_score(X_lda, X_train, y_train, cv=10, scoring = 'accuracy')
    # Prediction and performance on test set

    y_pred=X_lda.predict(X_test)
    test_acc=accuracy_score(y_pred, y_test)
    print('Train/test accuracy %.3f/%.3f'% (round(scores.mean(),2), round(test_acc.mean(),2)))

        
    return test_acc

def adaboo(X,y):
    print('Adaboost:')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, stratify=y)
    ada = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=5),
        n_estimators =40,
        learning_rate=0.6)

    ada = ada.fit(X_train, y_train)
    # The (10-fold) cross-validated score is found

    tree_train = cross_val_score(ada, X_train, y_train, cv=10, scoring = 'accuracy')
    # Prediction and performance on test set

    y_pred = ada.predict(X_test)
    tree_test = accuracy_score(y_pred, y_test)
    print('Train/test accuracies %.3f/%.3f'% (round(tree_train.mean(),2), tree_test))
    
    return tree_test

def classify(X, covar, method):
    if method ==0:
        X_res=RFC(X, covar)
        return X_res
    elif method==1:
        X_res=LSVC(X, covar)
        return X_res
    elif method==2:
        X_res=MLP_calc(X, covar)
        # X_res=0.1
        return X_res
    elif method==3:
        X_res=logis(X,covar)
        return X_res
    elif method==4:
        X_res=LDA_class(X,covar)
        return X_res
    elif method==5:
        X_res=adaboo(X,covar)
        return X_res
        
    return X_res
