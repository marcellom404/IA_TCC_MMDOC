import time
from IA_Aprendisado_de_maquina import treinar_todos, imprimir_resultados
from DATA_load import treinamento
import os

# para alimentar o processador de resultados
tam = 400
start_time = time.time()

for i in range(tam):
    iter_start_time = time.time()
    
    print(f"--- Iteração {i+1}/{tam} ---")

    resultados = treinar_todos(treinamento)
    os.system('cls')
    
    # imprimir_resultados(resultados)

    iter_end_time = time.time()
    iter_duration = iter_end_time - iter_start_time
    
    elapsed_time = iter_end_time - start_time
    progress = (i + 1) / tam
    avg_time_per_iter = elapsed_time / (i + 1)
    remaining_iter = tam - (i + 1)
    eta = remaining_iter * avg_time_per_iter

    print(f"Progresso: {progress:.2%}")
    print(f"Tempo decorrido: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
    print(f"Tempo estimado para conclusão: {time.strftime('%H:%M:%S', time.gmtime(eta))}")
    print(f"Duração desta iteração: {time.strftime('%H:%M:%S', time.gmtime(iter_duration))}")
    print("-" * 20)
    
