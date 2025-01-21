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
DATASET_NAME = 'CSECICIDS2018_improved/Friday-02-03-2018.csv'
print(path + "/" + DATASET_NAME)
# Carregando os dados em um DataFrame

num_linhas_total = 6168188  
num_linhas_desejado = 61681  
# Gera uma lista de índices de linhas para pular aleatoriamente pq meu pc nao tem memoria infinita ainda
skip_indices = sorted(random.sample(range(1, num_linhas_total + 1), 
                                     num_linhas_total - num_linhas_desejado))

df = pd.read_csv(path + "/" + DATASET_NAME,skiprows=skip_indices)


# colunas que precisam ser removidas
for i in df.columns.values,df.iloc[1] :
    print(i)
for i,j in list(zip(df.columns.values,df.iloc[1] )):
  print(i,"| De tipo:",str(type(j)).replace("<","").replace(">","").replace("class","").replace("64","").replace("numpy.","").replace("'","").replace(" ",""))
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

# Função para inicializar variáveis globais
def formatar_variaveis():
    global melhor_arvore, melhor_knn, melhor_mlp
    global saida_arvore, saida_mlp, saida_knn

    # Listas para armazenar saídas dos modelos
    saida_arvore = []
    saida_mlp = []
    saida_knn = []

    # Variáveis para armazenar o melhor resultado de cada modelo
    melhor_arvore = ["", 0]
    melhor_mlp = ["", 0]
    melhor_knn = ["", 0]
    treinamento_tempo = 0
# Inicializando variáveis antes de iniciar o processo
formatar_variaveis()

# Função para gerar gráficos com os resultados
def gerar_grafico(dados, nome):
    # Extraindo eixos x e y a partir dos dados fornecidos
    x = [d[0] for d in dados]
    y = [d[1] * 100 for d in dados]  # Convertendo precisão para porcentagem

    # Definindo o gráfico com barras e linha para melhor visualização
    plt.plot(x, y, linestyle='-', color='r', label='Linha de desempenho')
    plt.bar(x, y, color='b', alpha=0.6, label='Desempenho por parâmetro')

    # Configurando o eixo y para mostrar entre 50% e 100%
    plt.ylim(50, 100)

    # Definindo título e rótulos dos eixos
    plt.title(f"Desempenho do Modelo - {nome}")
    plt.xlabel("Parâmetros")
    plt.ylabel("Precisão (%)")

    # Exibindo a grade e a legenda para facilitar a leitura do gráfico
    plt.grid(True)
    plt.legend()

    # Mostrando o gráfico final
    plt.show()

# Função para ajustar e avaliar o k-NN com diferentes números de vizinhos
def knn(data):
    global melhor_knn, saida_knn
    for i in [1, 2, 4, 8, 16, 32, 64, 65, 66, 67, 68]:
        # Inicializando o modelo com 'i' vizinhos
        ia = KNeighborsClassifier(n_neighbors=i)
        ia.fit(data.drop(alvo, axis=1), data[alvo])
        valida = ia.predict(validacao.drop(alvo, axis=1))

        # Avaliação da precisão no conjunto de validação
        prec = accuracy_score(validacao[alvo], valida)
        saida_knn.append([f"{i}", prec])
        print("knn",i,prec)
        # Armazenando o melhor modelo
        if melhor_knn[1] < prec:
            melhor_knn = [f"{i}", prec, valida]

knn(treinamento)
print(f"Usando k-NN, o melhor número de vizinhos foi {melhor_knn[0]}, com precisão de {melhor_knn[1]*100:.2f}%")

# Função para ajustar e avaliar o MLP com diferentes quantidades de camadas ocultas
def mlp(data):
    global melhor_mlp, saida_mlp
    # Testando configurações de 2 a 8 camadas ocultas
    for i in [2, 3, 4, 5, 6, 7, 8]:
        # Inicializando o MLP com 'i' camadas ocultas e limite de iterações
        ia = MLPClassifier(hidden_layer_sizes=(i,), max_iter=1990)
        ia.fit(data.drop(alvo, axis=1), data[alvo])  # Treinamento do modelo

        # Realizando previsões no conjunto de validação
        valida = ia.predict(validacao.drop(alvo, axis=1))

        # Calculando a precisão
        prec = accuracy_score(validacao[alvo], valida)

        # Armazenando o número de camadas e precisão para análise posterior
        saida_mlp.append([f"{i}", prec])

        # Armazenando o melhor resultado encontrado
        if melhor_mlp[1] < prec:
            melhor_mlp = [f"{i}", prec, valida]

# Executando o ajuste do MLP no conjunto de treinamento
mlp(treinamento)

# Exibindo o resultado com o número ideal de camadas
print(f"O melhor resultado foi obtido com {melhor_mlp[0]} camadas ocultas, alcançando {melhor_mlp[1] * 100:.2f}% de precisão.")

# Função para ajustar e avaliar o modelo de Árvore de Decisão
def arvore(data):
    global melhor_arvore, saida_arvore
    # Testando profundidades da árvore variando entre 1 e 10
    for i in range(1, 11):
        # Inicializando a árvore de decisão
        ia = DecisionTreeClassifier(max_depth=i)
        ia.fit(data.drop(alvo, axis=1), data[alvo])  # Treinamento do modelo

        # Realizando previsões no conjunto de validação
        valida = ia.predict(validacao.drop(alvo, axis=1))

        # Avaliando a precisão
        prec = accuracy_score(validacao[alvo], valida)

        # Registrando resultados para análise posterior
        saida_arvore.append([f"{i}", prec])

        # Armazenando o melhor resultado
        if melhor_arvore[1] < prec:
            melhor_arvore = [f"{i}", prec, valida]

# Executando o ajuste da árvore de decisão no conjunto de treinamento
arvore(treinamento)

# Exibindo o resultado
print(f"O melhor resultado para a árvore de decisão foi obtido com profundidade {melhor_arvore[0]}, com precisão de {melhor_arvore[1] * 100:.2f}%")

# Função para treinar todos os modelos
def treinar_todos(dados):
    arvore(dados)
    knn(dados)
    mlp(dados)

# Inicializando variáveis antes de iniciar o processo
formatar_variaveis()

# Treinando todos os modelos com o conjunto de treinamento
treinar_todos(treinamento)  # O processo leva cerca de 15 segundos

# Exibindo as melhores precisões alcançadas pelos algoritmos
print("Estas são as melhores precisões alcançadas pelos algoritmos:")
print(f"MLP: {melhor_mlp[1]*100:.5f}%, Árvore de Decisões: {melhor_arvore[1]*100:.5f}%, e K-NN: {melhor_knn[1]*100:.5f}%")

# Identificando o modelo com a melhor precisão
melhor_modelo = ["", 0]
resultados = [
    ["MLP", melhor_mlp[1]],
    ["Árvore de Decisões", melhor_arvore[1]],
    ["K-NN", melhor_knn[1]]
]

for modelo in resultados:
    if modelo[1] > melhor_modelo[1]:
        melhor_modelo = modelo

print(f"O melhor foi o {melhor_modelo[0]}, que teve {melhor_modelo[1] * 100:.5f}% de acerto.")

formatar_variaveis()
treinar_todos(treinamento) # O treinamento leva cerca de 15s

gerar_grafico(saida_arvore, "Árvore de Decisão")
gerar_grafico(saida_knn, "K-NN")
gerar_grafico(saida_mlp, "Multilayer Perceptron")