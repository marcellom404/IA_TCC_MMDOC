import sqlite3
import json
from collections import defaultdict

def calculate_averages():
    """Lê o banco de dados, calcula a média de acurácia para cada algoritmo e exibe os resultados."""
    conn = sqlite3.connect('results.db')
    conn.row_factory = sqlite3.Row 
    cursor = conn.cursor()
    cursor.execute('SELECT algorithm, accuracy FROM results')
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("Nenhum resultado encontrado no banco de dados.")
        return

    # Agrupando as acurácias por algoritmo
    accuracies_by_algorithm = defaultdict(list)
    for row in rows:
        accuracies_by_algorithm[row['algorithm']].append(row['accuracy'])

    # Calculando e exibindo a média para cada algoritmo
    print("Média de acurácia por algoritmo:")
    for algorithm, accuracies in accuracies_by_algorithm.items():
        average_accuracy = sum(accuracies) / len(accuracies)
        print(f"  {algorithm}: {average_accuracy * 100:.4f}%")

if __name__ == '__main__':
    calculate_averages()
