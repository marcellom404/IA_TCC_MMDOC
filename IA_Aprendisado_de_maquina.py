import threading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import os
import manipul_arquivo as mov
import funcoes_teste_dataset as teste
import random
import  conversor as conv
import time
from DATA_load import df, alvo, get_dados_amostra
import database
import diagnostico

# Criar a tabela do banco de dados se não existir
database.create_table()

# adicionar outros metodos de avaliação, como o f1 score, e o recall, e o precision
# pandas,numpy,matplotlib,kagglehub,sklearn

# Função para gerar gráficos com os resultados
def gerar_grafico(dados, nome):
    # Cria uma nova figura para cada gráfico, para que não se sobreponham
    plt.figure()
    
    # Extraindo eixos x e y a partir dos dados fornecidos
    # plt.sybplot 
    x = [d[0] for d in dados]
    y = [d[1] * 100 for d in dados]  # Convertendo precisão para porcentagem

    # Definindo o gráfico com barras e linha para melhor visualização
    plt.plot(x, y, linestyle='-', color='r', label='Linha de desempenho')
    plt.bar(x, y, color='b', alpha=0.6, label='Desempenho por parâmetro')

    # Configurando o eixo y para mostrar do menor valor até 100%
    if y:  # Evita erro se a lista de dados estiver vazia
        plt.ylim(bottom=min(y) - 1, top=100.5)  # Adiciona margem visual

    # Definindo título e rótulos dos eixos
    plt.title(f"Desempenho do Modelo - {nome}")
    plt.xlabel("Parâmetros")
    plt.ylabel("Precisão (%)\n")

    # Exibindo a grade e a legenda para facilitar a leitura do gráfico
    plt.grid(True)
    plt.legend()

# Função para ajustar e avaliar o k-NN com diferentes números de vizinhos
def knn(data, validacao):
    saida_knn = []
    melhor_knn = ["", 0, "", 0, 0, 0, 0]
    for i in [1, 2, 4, 8, 16, 32, 64, 65, 66, 67, 68]:
        # Inicializando o modelo com 'i' vizinhos
        ia = KNeighborsClassifier(n_neighbors=i)
        ia.fit(data.drop(alvo, axis=1), data[alvo])
        
        start_time = time.time()
        valida = ia.predict(validacao.drop(alvo, axis=1))
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000 / len(validacao)

        # Avaliação da precisão no conjunto de validação
        prec = accuracy_score(validacao[alvo], valida)
        report = classification_report(validacao[alvo], valida, output_dict=True, zero_division=0)
        
        attack_metrics = {}
        for label, metrics in report.items():
            if label.upper() != 'BENIGN' and label not in ['accuracy', 'macro avg', 'weighted avg']:
                attack_metrics = metrics
                break

        precisao_ataque = attack_metrics.get('precision', 0)
        recall_ataque = attack_metrics.get('recall', 0)
        f1_score_ataque = attack_metrics.get('f1-score', 0)

        diagnostico.analisar_resultado('K-NN', ia, f1_score_ataque, attack_metrics, validacao, alvo)

        saida_knn.append([f"{i}", prec])
        
        # Armazenando o melhor modelo
        if melhor_knn[1] < prec:
            melhor_knn = [f"{i}", prec, valida, precisao_ataque, recall_ataque, f1_score_ataque, inference_time]
    
    # Salvar o melhor resultado no banco de dados
    database.save_result('K-NN', melhor_knn[1], melhor_knn[3], melhor_knn[4], melhor_knn[5], melhor_knn[6], {'n_neighbors': int(melhor_knn[0])})
    return melhor_knn, saida_knn


# Função para ajustar e avaliar o MLP com diferentes quantidades de camadas ocultas
def mlp(data, validacao):
    saida_mlp = []
    melhor_mlp = ["", 0, "", 0, 0, 0, 0]
    # Testando configurações de 2 a 8 camadas ocultas
    for i in [2, 3, 4, 5, 6, 7, 8]:
        # Inicializando o MLP com 'i' camadas ocultas e limite de iterações
        ia = MLPClassifier(hidden_layer_sizes=(i,), max_iter=1990)
        ia.fit(data.drop(alvo, axis=1), data[alvo])  # Treinamento do modelo

        start_time = time.time()
        # Realizando previsões no conjunto de validação
        valida = ia.predict(validacao.drop(alvo, axis=1))
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000 / len(validacao)

        # Calculando a precisão
        prec = accuracy_score(validacao[alvo], valida)
        report = classification_report(validacao[alvo], valida, output_dict=True, zero_division=0)

        attack_metrics = {}
        for label, metrics in report.items():
            if label.upper() != 'BENIGN' and label not in ['accuracy', 'macro avg', 'weighted avg']:
                attack_metrics = metrics
                break
        
        precisao_ataque = attack_metrics.get('precision', 0)
        recall_ataque = attack_metrics.get('recall', 0)
        f1_score_ataque = attack_metrics.get('f1-score', 0)

        diagnostico.analisar_resultado('MLP', ia, f1_score_ataque, attack_metrics, validacao, alvo)

        # Armazenando o número de camadas e precisão para análise posterior
        saida_mlp.append([f"{i}", prec])

        # Armazenando o melhor resultado encontrado
        if melhor_mlp[1] < prec:
            melhor_mlp = [f"{i}", prec, valida, precisao_ataque, recall_ataque, f1_score_ataque, inference_time]

    # Salvar o melhor resultado no banco de dados
    database.save_result('MLP', melhor_mlp[1], melhor_mlp[3], melhor_mlp[4], melhor_mlp[5], melhor_mlp[6], {'hidden_layer_sizes': int(melhor_mlp[0])})
    return melhor_mlp, saida_mlp

def random_forest(data, validacao):
    saida_random_forest = []
    melhor_random_forest = ["", 0, "", 0, 0, 0, 0]
    # Testando com diferentes números de árvores na floresta
    for n_arvores in range(15,25):
        ia = RandomForestClassifier(n_estimators=n_arvores, random_state=42, n_jobs=-1) # n_jobs=-1 usa todos os cores do processador
        ia.fit(data.drop(alvo, axis=1), data[alvo])
        
        start_time = time.time()
        valida = ia.predict(validacao.drop(alvo, axis=1))
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000 / len(validacao)

        prec = accuracy_score(validacao[alvo], valida)
        report = classification_report(validacao[alvo], valida, output_dict=True, zero_division=0)

        attack_metrics = {}
        for label, metrics in report.items():
            if label.upper() != 'BENIGN' and label not in ['accuracy', 'macro avg', 'weighted avg']:
                attack_metrics = metrics
                break

        precisao_ataque = attack_metrics.get('precision', 0)
        recall_ataque = attack_metrics.get('recall', 0)
        f1_score_ataque = attack_metrics.get('f1-score', 0)

        diagnostico.analisar_resultado('Random Forest', ia, f1_score_ataque, attack_metrics, validacao, alvo)

        saida_random_forest.append([f"{n_arvores} árvores", prec])
        # print(f"Random Forest com {n_arvores} árvores: {prec*100:.5f}% de precisão.")
        if prec > melhor_random_forest[1]:
            melhor_random_forest = [f"{n_arvores} árvores", prec, valida, precisao_ataque, recall_ataque, f1_score_ataque, inference_time]

    # Salvar o melhor resultado no banco de dados
    # Extrai o número de árvores do string
    n_arvores_melhor = int(melhor_random_forest[0].split()[0])
    database.save_result('Random Forest', melhor_random_forest[1], melhor_random_forest[3], melhor_random_forest[4], melhor_random_forest[5], melhor_random_forest[6], {'n_estimators': n_arvores_melhor})
    return melhor_random_forest, saida_random_forest

# Função para treinar todos os modelos
def treinar_todos(dados, validacao):
    melhor_knn, saida_knn = knn(dados, validacao)
    melhor_mlp, saida_mlp = mlp(dados, validacao)
    melhor_random_forest, saida_random_forest = random_forest(dados, validacao)
    
    resultados = {
        "melhor_knn": melhor_knn,
        "saida_knn": saida_knn,
        "melhor_mlp": melhor_mlp,
        "saida_mlp": saida_mlp,
        "melhor_random_forest": melhor_random_forest,
        "saida_random_forest": saida_random_forest
    }
    return resultados

def imprimir_resultados(resultados):
    melhor_mlp = resultados["melhor_mlp"]
    melhor_knn = resultados["melhor_knn"]
    melhor_random_forest = resultados["melhor_random_forest"]

    print("Estas são as melhores precisões alcançadas pelos algoritmos:")
    print(f'''MLP: {melhor_mlp[1]*100:.5f}% Erra uma ves a cada {conv.calcular_frequencia_de_erro(melhor_mlp[1]*100)} tentativas \n\
K-NN: {melhor_knn[1]*100:.5f}% Erra uma ves a cada {conv.calcular_frequencia_de_erro(melhor_knn[1]*100)} tentativas \n\
Random Forest: {melhor_random_forest[1]*100:.5f}% Erra uma ves a cada {conv.calcular_frequencia_de_erro(melhor_random_forest[1]*100)} tentativas''')

    # Identificando o modelo com a melhor precisão
    melhor_modelo = ["", 0]
    lista_resultados = [
        ["MLP", melhor_mlp[1]],
        ["K-NN", melhor_knn[1]],
        ["Random Forest", melhor_random_forest[1]]
    ]

    for modelo in lista_resultados:
        if modelo[1] > melhor_modelo[1]:
            melhor_modelo = modelo
    
    print(f"O melhor foi o {melhor_modelo[0]}, que teve {melhor_modelo[1] * 100:.5f}% de acerto.")

if __name__ == '__main__':
    treinamento, validacao, teste = get_dados_amostra()
    resultados = treinar_todos(treinamento, validacao)
    imprimir_resultados(resultados)

    saida_knn = resultados["saida_knn"]
    saida_mlp = resultados["saida_mlp"]
    saida_random_forest = resultados["saida_random_forest"]
    
    gerar_grafico(saida_knn, "K-NN")
    gerar_grafico(saida_mlp, "Multilayer Perceptron")
    gerar_grafico(saida_random_forest, "Random Forest")

    # Exibe todas os graficos criadas de uma só vez, cada uma em sua janela.
    plt.show()