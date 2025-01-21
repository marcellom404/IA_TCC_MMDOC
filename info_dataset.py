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
# minha nossa como demorou rodar isso, no final tem um comentario com a saida, e so pra saber quantos e os tipos pacotes dos datasets


if not os.path.exists("datasets"):
    print("--")
    path = mov.get_dataset_paths_from_db()[0]
elif os.path.exists("datasets"):
    path = "datasets/1"
else:
    path = kagglehub.dataset_download("ernie55ernie/improved-cicids2017-and-csecicids2018")
    mov.save_dataset_path_to_db(mov.move_dataset(path))
# 'CSECICIDS2018_improved/Friday-02-03-2018.csv' botnet
# 'CSECICIDS2018_improved/Friday-16-02-2018.csv' DOS hulk+ftp-bruteforce
dataset_names = os.listdir("datasets\\1\\CSECICIDS2018_improved") #terminar de colocar os nomes aqui
for i in dataset_names:
    DATASET_NAME = i
    print(path + "/CSECICIDS2018_improved/" + DATASET_NAME)
    # Carregando os dados em um DataFrame


    df = pd.read_csv(path + "/CSECICIDS2018_improved/" + DATASET_NAME)

    # como em algums dos datasets tem quantidades muito desproporcionais de dados(90% ser benigno por exemplo) vamos deixar mais homogêneo

    # colunas que precisam ser removidas
    # for i in df.columns.values,df.iloc[1] :
    #     print(i)

    # print("----------------------------------------",teste.encontrar_colunas_com_valor_especifico(df,alvo,"BENIGN"))

    # "label" e a coluna que indica o tipo de fluxo de rede, sendo BENIGN o trafego comum, outros sao algum tipo de tentativa de intrusão
    print(df["Label"].value_counts())
    
# datasets/1/CSECICIDS2018_improved/Friday-02-03-2018.csv
# Label
# BENIGN                     6168188
# Botnet Ares                 142921
# Botnet Ares - Attempted        262
# Name: count, dtype: int64
# datasets/1/CSECICIDS2018_improved/Friday-16-02-2018.csv
# Label
# BENIGN                        5481500
# DoS Hulk                      1803160
# FTP-BruteForce - Attempted     105520
# DoS Hulk - Attempted               86
# Name: count, dtype: int64
# datasets/1/CSECICIDS2018_improved/Friday-23-02-2018.csv
# Label
# BENIGN                                  5976251
# Web Attack - XSS                             73
# Web Attack - Brute Force                     62
# Web Attack - Brute Force - Attempted         61
# Web Attack - SQL                             23
# Web Attack - SQL - Attempted                 10
# Web Attack - XSS - Attempted                  1
# Name: count, dtype: int64
# datasets/1/CSECICIDS2018_improved/Thursday-01-03-2018.csv
# Label
# BENIGN                                          6511554
# Infiltration - NMAP Portscan                      39634
# Infiltration - Communication Victim Attacker        161
# Infiltration - Dropbox Download                      39
# Infiltration - Dropbox Download - Attempted          13
# Name: count, dtype: int64
# datasets/1/CSECICIDS2018_improved/Thursday-15-02-2018.csv
# Label
# BENIGN                       5372471
# DoS GoldenEye                  22560
# DoS Slowloris                   8490
# DoS GoldenEye - Attempted       4301
# DoS Slowloris - Attempted       2280
# Name: count, dtype: int64
# datasets/1/CSECICIDS2018_improved/Thursday-22-02-2018.csv
# Label
# BENIGN                                  6070945
# Web Attack - Brute Force - Attempted         76
# Web Attack - Brute Force                     69
# Web Attack - XSS                             40
# Web Attack - SQL                             16
# Web Attack - SQL - Attempted                  4
# Web Attack - XSS - Attempted                  3
# Name: count, dtype: int64
# datasets/1/CSECICIDS2018_improved/Tuesday-20-02-2018.csv
# Label
# BENIGN                       5764497
# DDoS-LOIC-HTTP                289328
# DDoS-LOIC-UDP                    797
# DDoS-LOIC-UDP - Attempted         80
# Name: count, dtype: int64
# datasets/1/CSECICIDS2018_improved/Wednesday-14-02-2018.csv
# FTP-BruteForce - Attempted     193354
# SSH-BruteForce                  94197
# Name: count, dtype: int64
# datasets/1/CSECICIDS2018_improved/Wednesday-21-02-2018.csv
# Label
# BENIGN                       5878399
# DDoS-HOIC                    1082293
# DDoS-LOIC-UDP                   1730
# DDoS-LOIC-UDP - Attempted        171
# Name: count, dtype: int64
# datasets/1/CSECICIDS2018_improved/Wednesday-28-02-2018.csv
# Label
# BENIGN                                          6518882
# Infiltration - NMAP Portscan                      49740
# Infiltration - Dropbox Download                      46
# Infiltration - Communication Victim Attacker         43
# Infiltration - Dropbox Download - Attempted          15