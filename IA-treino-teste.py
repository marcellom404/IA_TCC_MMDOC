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
# Caminho para o dataset




if os.path.exists(mov.get_dataset_paths_from_db()[0]):
    path = mov.get_dataset_paths_from_db()[0]
else:
    
    path = kagglehub.dataset_download("ernie55ernie/improved-cicids2017-and-csecicids2018")
    mov.save_dataset_path_to_db(mov.move_dataset(path))
# 'CSECICIDS2018_improved/Friday-02-03-2018.csv' botnet
# 'CSECICIDS2018_improved/Friday-16-02-2018.csv' DOS hulk+ftp-bruteforce
DATASET_NAME = 'CSECICIDS2018_improved/Friday-02-03-2018.csv'
print(path + "/" + DATASET_NAME)
# Carregando os dados em um DataFrame
df = pd.read_csv(path + "/" + DATASET_NAME, nrows=10000)

# "label" e a coluna que indica o tipo de fluxo de rede, sendo BENIGN o trafego comum, outros sao algum tipo de tentativa de intrusão
print(df["Label"].value_counts())

treinamento = df.sample(frac=0.7, random_state=64)
df_restante = df.drop(treinamento.index)
validacao = df_restante.sample(frac=0.6667, random_state=64)
teste = df_restante.drop(validacao.index)

print("Total de 4601 linhas no conjunto original, incluindo uma linha com nomes das colunas.")
print(f"Validação: {len(validacao)} linhas")
print(f"Treinamento: {len(treinamento)} linhas")
print(f"Teste: {len(teste)} linhas")
print("Soma total:", len(validacao) + len(treinamento) + len(teste))
