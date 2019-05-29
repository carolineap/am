import numpy as np
import pandas as pd
import svmutil
from svmutil import svm_read_problem
from svmutil import svm_problem
from svmutil import svm_parameter
from svmutil import svm_train
from svmutil import svm_predict
from svmutil import svm_save_model

def gridSearch(X, Y, Xval, Yval):
    #inicializa as variáveis que deverão ser retornadas pela função
    custo = 1000
    gamma = 1000
    
    values = []
    for i in range(1, 10):
         for j in range(-1, 3):
            values.append(i/(10 ** j))
    
    acr = 0
    
    for i in values:
        for j in values:
            model = svm_train(Y, X, '-c %f -g %f -q' %(i, j))
            classes = svm_predict([], Xval, model, '-q')
            acuracia = np.sum(np.array(classes[0])==Yval)/len(Yval)
            if acr < acuracia:
                custo = i
                gamma = j
                acr = acuracia
    
    return custo, gamma

def svm(Xtrain,Ytrain,Xval,Yval) :

    bestC, bestGamma = gridSearch(Xtrain, Ytrain, Xval, Yval)

    kernel = 2 #kernel radial
    model = svm_train(Ytrain, Xtrain, '-c %f -t %d -g %f -q' %(bestC, kernel, bestGamma))
    classes = svm_predict([], Xval, model, '-q')
    acuracia = np.sum(np.array(classes[0])==Yval)/len(Yval)

    return acuracia


    