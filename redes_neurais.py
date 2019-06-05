import numpy as np
import pandas as pd
import scipy.optimize









<<<<<<< HEAD
    # Qtde de amostras
    m = X.shape[0]
         
    #Custo
    J = 0
    
    #Theta das redes neurais
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    
    eps = 1e-15

    #
    rotulos = np.zeros((len(y), num_labels), dtype=int) 
    for i in range(len(rotulos)):
        rotulos[i][y[i]] = 1

    a1 = np.insert(X, 0, np.ones(m, dtype=int), axis=1)
    z2 = np.dot(a1, Theta1.T)
    a2 = sigmoid(z2)
    a2 = np.insert(a2, 0, np.ones(a2.shape[0], dtype=int), axis=1)
    z3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(z3)
    
    
    delta3 = a3 - rotulos
    delta2 = np.dot(delta3, Theta2[:, 1:]) * sigmoidGradient(z2)

    if flagreg == True :
        reg = (lambda_reg/(2*m))*(np.sum(Theta1[:, 1:] ** 2) + np.sum(Theta2[:, 1:] ** 2)) 

        J += reg
    
        delta3 = a3 - rotulos
        delta2 = np.dot(delta3, Theta2[:, 1:]) * sigmoidGradient(z2)

        Theta1_grad[:, 0] = (np.dot(delta2.T, a1)[:, 0]/m) 
        Theta1_grad[:, 1:] = (np.dot(delta2.T, a1)[:, 1:]/m) + (lambda_reg/m)*Theta1[:, 1:]

        Theta2_grad[:, 0] = (np.dot(delta3.T, a2)[:, 0]/m) 
        Theta2_grad[:, 1:] = (np.dot(delta3.T, a2)[:, 1:]/m) + (lambda_reg/m)*Theta2[:, 1:]
    else :
        Theta1_grad = np.dot(delta2.T, a1)/m
        Theta2_grad = np.dot(delta3.T, a2)/m
     
    J = np.sum(-rotulos*np.log(a3+eps) - (1 - rotulos)*np.log(1 - a3+eps))/m

    grad = np.concatenate([np.ravel(Theta1_grad), np.ravel(Theta2_grad)])

    return J, grad


def predicao(Theta1, Theta2, X):

    m = X.shape[0] # número de amostras
    num_labels = Theta2.shape[0]
    
    p = np.zeros(m)

    a1 = np.hstack( [np.ones([m,1]),X] )
    h1 = sigmoid( np.dot(a1,Theta1.T) )

    a2 = np.hstack( [np.ones([m,1]),h1] ) 
    h2 = sigmoid( np.dot(a2,Theta2.T) )
    
    p = np.argmax(h2,axis=1)
    
    return p

def redes_neurais (Xtrain, Ytrain,Xval,Yval):

    input_layer_size  = Xtrain.shape[1]  # 20x20 dimensao das imagens de entrada
    hidden_layer_size = int((input_layer_size + 2)*2/3)   # 25 neuronios na camada oculta
    num_labels = 2          # 10 rotulos, de 1 a 10  
                         #  (observe que a classe "0" recebe o rotulo 10)
    epsilon_init = 0.12
    
    # carregando os pesos da camada 1
    Theta1 =  np.random.RandomState(10).rand(hidden_layer_size, 1 + input_layer_size) * 2 * epsilon_init - epsilon_init
    
    # carregando os pesos da camada 2
    Theta2 = np.random.RandomState(10).rand(num_labels, 1 + hidden_layer_size) * 2 * epsilon_init - epsilon_init
    
    # concatena os pesos em um único vetor
    initial_rna_params = np.concatenate([np.ravel(Theta1), np.ravel(Theta2)])
    
    MaxIter = 5000
    
    lamba_reg = 1
    
    # Minimiza a funcao de custo
    result = scipy.optimize.minimize(fun=funcaoCusto_backp, x0=initial_rna_params, args=(input_layer_size, hidden_layer_size, num_labels, Xtrain, Ytrain,True,lamba_reg),  
                    method='TNC', jac=True, options={'maxiter': MaxIter})
    
    # Coleta os pesos retornados pela função de minimização
    nn_params = result.x
    
    # Obtem Theta1 e Theta2 back a partir de rna_params
    Theta1 = np.reshape( nn_params[0:hidden_layer_size*(input_layer_size + 1)], (hidden_layer_size, input_layer_size+1) )
    Theta2 = np.reshape( nn_params[ hidden_layer_size*(input_layer_size + 1):], (num_labels, hidden_layer_size+1) )
    
    pred = predicao(Theta1, Theta2, Xval)
    
    return np.mean( pred == Yval ) 
    
=======

>>>>>>> c44560ec362353cae5da15c08f6c2a9d704dbc05
