from IA_Aprendisado_de_maquina import treinar_todos, imprimir_resultados, gerar_grafico
from DATA_load import treinamento
import matplotlib.pyplot as plt

# Treinando todos os modelos com o conjunto de treinamento
resultados = treinar_todos(treinamento)

# Imprimindo os resultados
imprimir_resultados(resultados)

# Gerando os gráficos
gerar_grafico(resultados["saida_arvore"], "Árvore de Decisão")
gerar_grafico(resultados["saida_knn"], "K-NN")
gerar_grafico(resultados["saida_mlp"], "Multilayer Perceptron")
gerar_grafico(resultados["saida_random_forest"], "Random Forest")

# Exibe todas os graficos criadas de uma só vez, cada uma em sua janela.
plt.show()