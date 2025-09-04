import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from DATA_load import treinamento, validacao, alvo, teste
import conversor as conv

# Variáveis globais para armazenar os melhores resultados
melhor_random_forest = ["", 0]
melhor_svc = ["", 0]

# Função para avaliar o RandomForest com diferentes números de árvores
def random_forest(data):
    global melhor_random_forest
    # Testando com diferentes números de árvores na floresta
    for n_arvores in [50, 100, 200]:
        ia = RandomForestClassifier(n_estimators=n_arvores, random_state=42, n_jobs=-1) # n_jobs=-1 usa todos os cores do processador
        ia.fit(data.drop(alvo, axis=1), data[alvo])
        valida = ia.predict(validacao.drop(alvo, axis=1))
        prec = accuracy_score(validacao[alvo], valida)
        
        if prec > melhor_random_forest[1]:
            melhor_random_forest = [f"{n_arvores} árvores", prec]



# Treinando os novos modelos
print("Iniciando treinamento dos algoritmos adicionais...")
print("Treinando RandomForest...")
random_forest(treinamento)
print("Treinamento do RandomForest concluído.")



# Exibindo os melhores resultados
print("\n--- Resultados dos Algoritmos Adicionais ---")
print(f"Random Forest: Melhor resultado com {melhor_random_forest[0]}, precisão de {melhor_random_forest[1]*100:.5f}%. Erra uma vez a cada {conv.calcular_frequencia_de_erro(melhor_random_forest[1]*100)} tentativas.")
