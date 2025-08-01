
import os
import manipul_arquivo as mov
import funcoes_teste_dataset as teste
import random
import pandas as pd
import kagglehub

global alvo 
alvo = "Label"

if os.path.exists("datasets"):

    path = "datasets/1"
else:
    path = kagglehub.dataset_download("ernie55ernie/improved-cicids2017-and-csecicids2018")
    #move o dataset para fora dos arquivos temporarios, salvando em datasets/
    mov.save_dataset_path_to_db(mov.move_dataset(path))
# informaçoes sobre os datasets no info_dataset.py 'CSECICIDS2018_improved/Friday-02-03-2018.csv'
DATASET_NAME = "thursday.csv"
print(path + "/" + DATASET_NAME)
# Carregando os dados em um DataFrame

num_linhas_total = 6168188  
num_linhas_desejado = 61681  
# Gera uma lista de índices de linhas para pular aleatoriamente pq meu pc nao tem memoria infinita... ainda
skip_indices = sorted(random.sample(range(1, num_linhas_total + 1), 
                                     num_linhas_total - num_linhas_desejado))
# remova o "skiprows=skip_indices" abaixo se for usar o dataset inteiro
df = pd.read_csv(path + "/" + DATASET_NAME,skiprows=skip_indices)


# mostra as colunas
for i in df.columns.values,df.iloc[1] :
    print(i)
for i,j in list(zip(df.columns.values,df.iloc[1] )):
  print(i,"| De tipo:",str(type(j)).replace("<","").replace(">","").replace("class","").replace("64","").replace("numpy.","").replace("'","").replace(" ",""))
# colunas que precisam ser removidas
columns_to_drop = ["id", "Attempted Category", "Timestamp", "Src IP", "Dst IP","Flow ID","Fwd URG Flags",'Bwd URG Flags', 'URG Flag Count']
df.drop(columns=columns_to_drop, inplace=True)
# print("----------------------------------------",teste.encontrar_colunas_com_valor_especifico(df,alvo,"BENIGN"))

# "label" e a coluna que indica o tipo de fluxo de rede, sendo BENIGN o trafego comum, outros sao algum tipo de tentativa de intrusão
print(df["Label"].value_counts())

treinamento = df.sample(frac=0.7, random_state=64)
df_restante = df.drop(treinamento.index)
validacao = df_restante.sample(frac=0.6667, random_state=64)
teste = df_restante.drop(validacao.index).sample(frac=0.01,random_state=64)

print("Total de 4601 linhas no conjunto original, incluindo uma linha com nomes das colunas.")
print(f"Validação: {len(validacao)} linhas")
print(f"Treinamento: {len(treinamento)} linhas")
print(f"Teste: {len(teste)} linhas")
print("Soma total:", len(validacao) + len(treinamento) + len(teste))