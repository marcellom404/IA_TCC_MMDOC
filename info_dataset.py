import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import os
import manipul_arquivo as mov
import funcoes_teste_dataset as teste
import random
global alvo 
alvo = "Label"



if os.path.exists(mov.get_dataset_paths_from_db()[0]):
    path = mov.get_dataset_paths_from_db()[0]
else:
    
    path = kagglehub.dataset_download("ernie55ernie/improved-cicids2017-and-csecicids2018")
    mov.save_dataset_path_to_db(mov.move_dataset(path))
# 'CSECICIDS2018_improved/Friday-02-03-2018.csv' botnet
# 'CSECICIDS2018_improved/Friday-16-02-2018.csv' DOS hulk+ftp-bruteforce
dataset_names = ["Friday-02-03-2018.csv","Friday-16-02-2018.csv"] #terminar de colocar os nomes aqui
for i in dataset_names:
    DATASET_NAME = i
    print(path + "/" + DATASET_NAME)
    # Carregando os dados em um DataFrame


    df = pd.read_csv(path + "/" + DATASET_NAME)

    # como em algums dos datasets tem quantidades muito desproporcionais de dados(90% ser benigno por exemplo) vamos deixar mais homogêneo

    # colunas que precisam ser removidas
    # for i in df.columns.values,df.iloc[1] :
    #     print(i)

    # print("----------------------------------------",teste.encontrar_colunas_com_valor_especifico(df,alvo,"BENIGN"))

    # "label" e a coluna que indica o tipo de fluxo de rede, sendo BENIGN o trafego comum, outros sao algum tipo de tentativa de intrusão
    print(df["Label"].value_counts())