import numpy as np 
import pandas as pd 

def normalizar(X):

    m, n = X.shape # m = qtde de objetos e n = qtde de atributos por objeto
    
    # Inicializa as variaves de saída
    X_norm = np.zeros( (m,n) ) #inicializa X_norm (base normalizada)
    mu = 0 # inicializa a média
    sigma = 1 # inicializa o desvio padrão
      
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=1)
    
    for i in range(m):
        for j in range(n):
            X_norm[i][j] = (X[i][j] - mu[j])/(sigma[j] + 0.000000001)
        
    return X_norm, mu, sigma

def distancia(x, X):
   
    m = X.shape[0] 
    D = np.zeros(m) 

    i = 0
    for amostra in X:
        D[i] = np.linalg.norm(amostra - x)
        i += 1
            
    return D

def knn(x, X, Y, K):
    
    X, mu, sigma = normalizar(X)
    
    y = 0 
    
    ind_viz = np.ones(K, dtype=int)
    
    D = distancia(x, X)
    
    votos = np.zeros(len(set(Y)))

    ind_viz = np.argsort(D)[:K]
            
    for indice in ind_viz:
        votos[Y[indice]] += 1    
            
    y = np.argmax(votos)
        
    return y


