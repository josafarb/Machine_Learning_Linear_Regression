#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 23:45:54 2020

@author: josafa
"""

import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
dados = pd.read_csv('500_Person_Gender_Height_Weight_Index.csv')


HOMEM = 'Male'
MULHER = 'Female'


EXRETMAMENTE_MAGRO = 0
MAGRO = 1
NORMAL = 2
ACIMA_DO_PESO = 3
OBESO = 4
EXTREMAMENTE_OBESO = 5

# Varíavel independente (x), nesse caso é o altura
VARIAVEL_X = 190

#Filtrando registros por valor de uma determinada coluna
dados = dados[dados.Gender ==  HOMEM]
dados = dados[dados.Index == NORMAL]

#Agrupando valores e relizando a média dos mesmos
dados = dados.groupby('Height',as_index=False).mean()

#Ordenando valores por altura
dados = dados.sort_values('Height')

#Definindo a varíavel independente(x) e a dependente (y)
x = dados[['Height']]
y = dados[['Weight']]

#dividindo o dataset em dados para treino e teste
x_train , x_teste, y_train, y_teste = train_test_split(x,y,test_size=0.2)

# Criando um 'novo' x, para realizar a previsão com o valor que eu passar
x_new = np.array([[VARIAVEL_X]])


#Definindo o linear regression como algorítmo utilizado
lr = linear_model.LinearRegression()

#Treinando modelo
lr.fit(x_train, y_train)

#Realizando previsão, tabém poderia passar no parâmetro os dados de teste 'x_teste'
y_pred = lr.predict(x_new)

#Imprimindo previsão
print(f'previsão {y_pred}')

plt.scatter(x_teste,y_teste)
#plt.plot(x_teste, y_pred, color='red', linewidth=3)
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()

#Verificando acurácia da previsão
print(lr.score(x_teste,y_teste))
input()
