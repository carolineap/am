import numpy as np 
import pandas as pd 

def calcularProbabilidades(X, Y):
   
    pPos = np.zeros(X.shape[1])
    pNeg = np.zeros(X.shape[1])

    pPos = X[(Y == 1)].sum(axis=0)/sum(Y == 1)
    pNeg = X[(Y == 0)].sum(axis=0)/sum(Y == 0)
    
    probPos = sum(Y==1)/len(Y) 
    probNeg = sum(Y==0)/len(Y)
    
    return probPos, probNeg, pPos, pNeg


def classificacao(x,probP,probN,pPos,pNeg):

    classe = 0;
    probPos= 0;
    probNeg = 0;


    probPos = probP * (pPos[(x != 0)]).prod() * (1 - pPos[(x == 0)]).prod()
    probNeg = probN * (pNeg[(x != 0)]).prod() * (1 - pNeg[(x == 0)]).prod()
    
    classe = int(probPos >= probNeg)

    return classe


