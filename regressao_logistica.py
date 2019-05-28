import numpy as np
import pandas as pd

def sigmoid(z) :

    if isinstance(z, int):
        g = 0
    else :
        g = np.zeros(z.shape)
    
    #Função sigmoidal
    g = 1/(1+ np.exp(-z))

    return g

def funcaoCusto(theta, X, Y, reg=0,lambda_reg=1) :
    
    m = len(Y)

    #Custo 
    J = 0
    #
    grad = np.zeros(len(theta))

    #Parâmetro de tolerância para a função de sigmoide 
    eps = 1e-15

    #Calcula a hipotese 
    h = sigmoid(np.dot(X, theta))
    
    J = (-Y * np.log(h + eps)) - ((1-Y)*np.log(1- h +eps))
    J = (1/m) * np.sum(J)

    grad = (1/m) * np.dot(X.T, h - Y) 

    if reg == 1 :
        sum_theta = np.sum(np.power(theta[1:],2))
        J = J + (lambda_reg/(2*m)) * (sum_theta)
        grad = grad + np.insert(((lambda_reg / m) * theta[1:]),0,0) 
       
    return J, grad

def predicao(theta, X) :

    m = X.shape[0]

    p = np.zeros(m, dtype=int)

    prob = sigmoid(np.dot(X, theta))
    p = (prob >= 0.5).astype(int) 

    return p


