import numpy as np
import pandas as pd


def stratified_kfolds(target, k, classes):
    #target é o vetor com as classes dos dados

    folds_final = np.zeros( k,dtype='object')

    # inicializa o vetor onde o k-esimo elemento guarda os indices dos dados de treinamento 
    # relativos ao k-esimo fold 
    train_index = np.zeros( k,dtype='object')
    
    # inicializa o vetor onde o k-esimo elemento guarda os indices dos dados de teste 
    # relativos ao k-esimo fold 
    test_index = np.zeros( k,dtype='object')
    
    # inicializa cada posicao do vetor folds_final que devera ser retornado pela funcao
    for i in folds_final:
        
        train_index[i] = [] # indices dos dados de treinamento relativos ao fold i
        test_index[i] = [] # indices dos dados de teste relativos ao fold i
        
        # inicializa o i-esimo elemento do vetor que devera ser retornado
        folds_final[i] = np.array( [train_index[i],test_index[i]] ) 
    
    folds = [[] for i in range(k)]
    
    for c in classes:
        index, = np.where(target == c)
        split_array = np.array_split(index, k)
        for i in range(k):
            folds[i].extend(split_array[i])
    
    test = []
    
    for i in range(k):
        
        test = folds[i]
        train = []
        
        for fold in folds:
            if fold != test:
                train.extend(fold)
    
        folds_final[i] = [train, test]
    
    return folds_final

def mediaFolds( resultados, classes ):
    
    nClasses = len(classes)
    
    acuracia = np.zeros( len(resultados) )

    revocacao = np.zeros( [len(resultados),len(classes)] )
    precisao = np.zeros( [len(resultados),len(classes)] )
    fmedida = np.zeros( [len(resultados),len(classes)] )

    revocacao_macroAverage = np.zeros( len(resultados) )
    precisao_macroAverage = np.zeros( len(resultados) )
    fmedida_macroAverage = np.zeros( len(resultados) )

    revocacao_microAverage = np.zeros( len(resultados) )
    precisao_microAverage = np.zeros( len(resultados) )
    fmedida_microAverage = np.zeros( len(resultados) )


    for i in range(len(resultados)):
        acuracia[i] = resultados[i]['acuracia']
        
        revocacao[i,:] = resultados[i]['revocacao']
        precisao[i,:] = resultados[i]['precisao']
        fmedida[i,:] = resultados[i]['fmedida']

        revocacao_macroAverage[i] = resultados[i]['revocacao_macroAverage']
        precisao_macroAverage[i] = resultados[i]['precisao_macroAverage']
        fmedida_macroAverage[i] = resultados[i]['fmedida_macroAverage']

        revocacao_microAverage[i] = resultados[i]['revocacao_microAverage']
        precisao_microAverage[i] = resultados[i]['precisao_microAverage']
        fmedida_microAverage[i] = resultados[i]['fmedida_microAverage']
    # imprimindo os resultados para cada classe
    print('\n\tRevocacao   Precisao   F-medida   Classe')
    for i in range(0,nClasses):
        print('\t%1.3f       %1.3f      %1.3f      %s' % (np.mean(revocacao[:,i]), np.mean(precisao[:,i]), np.mean(fmedida[:,i]), classes[i] ) )

    print('\t---------------------------------------------------------------------')
  
    #imprime as medias
    print('\t%1.3f       %1.3f      %1.3f      Média macro' % (np.mean(revocacao_macroAverage), np.mean(precisao_macroAverage), np.mean(fmedida_macroAverage)) )
    print('\t%1.3f       %1.3f      %1.3f      Média micro\n' % (np.mean(revocacao_microAverage), np.mean(precisao_microAverage), np.mean(fmedida_microAverage)) )

    print('\tAcuracia: %1.3f' %np.mean(acuracia))