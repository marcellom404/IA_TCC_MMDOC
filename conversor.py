def calcular_frequencia_de_erro(porcentagem_acerto: float) -> str:
    """
    Converte uma porcentagem de acerto em uma frequência de erro.

    Argumentos:
        porcentagem_acerto (float): A porcentagem de acerto do modelo (ex: 99.5).

    Retorna:
        numero de acertos para a ocorrência de 1 erro
    """
    # Validação da entrada
    if not 0 <= porcentagem_acerto <= 100:
        return "Erro: A porcentagem de acerto deve estar entre 0 e 100."

    # Caso especial: 100% de acerto significa que o modelo nunca erra.
    if porcentagem_acerto == 100:
        return "O modelo tem 100%% de acerto e, teoricamente, nunca erra."

    # 1. Calcular a porcentagem de erro
    taxa_erro_percentual = 100 - porcentagem_acerto

    # 2. Converter a porcentagem de erro para um valor decimal
    taxa_erro_decimal = taxa_erro_percentual / 100

    # 3. Calcular "1 em X" tentativas (o inverso da taxa de erro decimal)
    tentativas_para_um_erro = 1 / taxa_erro_decimal

    # 4. Arredondar para o número inteiro mais próximo para facilitar a leitura
    valor_x = round(tentativas_para_um_erro)
    
    

    return valor_x

# # --- Exemplos de uso com os seus dados ---

# mlp = 98.65952
# arvore_decisoes = 99.32976
# knn = 99.73190

# # Executando a função para cada modelo
# print(f"MLP ({mlp}%): {calcular_frequencia_de_erro(mlp)}")
# print(f"Árvore de Decisões ({arvore_decisoes}%): {calcular_frequencia_de_erro(arvore_decisoes)}")
# print(f"K-NN ({knn}%): {calcular_frequencia_de_erro(knn)}")

# # Exemplo com 100% de acerto
# print(f"Modelo Perfeito (100%): {calcular_frequencia_de_erro(100)}")

# # Exemplo com 50% de acerto
# print(f"Modelo Aleatório (50%): {calcular_frequencia_de_erro(50)}")