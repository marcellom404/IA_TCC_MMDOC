import sqlite3
import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def get_data_from_db():
    """Lê todos os resultados do banco de dados e retorna como um DataFrame do pandas."""
    if not os.path.exists('results.db'):
        print("Banco de dados 'results.db' não encontrado.")
        return None
    
    conn = sqlite3.connect('results.db')
    # Usar pandas para ler a query diretamente em um DataFrame
    try:
        df = pd.read_sql_query("SELECT * FROM results", conn)
    except pd.io.sql.DatabaseError:
        print("A tabela 'results' parece estar vazia ou com erro.")
        return None
    finally:
        conn.close()
        
    # Converter a string JSON de parâmetros para colunas separadas
    if 'parameters' in df.columns:
        parameters_df = df['parameters'].apply(lambda x: json.loads(x)).apply(pd.Series)
        df = pd.concat([df.drop('parameters', axis=1), parameters_df], axis=1)
        
    return df

def calculate_and_print_averages(df):
    """Calcula e exibe as médias das principais métricas para cada algoritmo."""
    if df is None or df.empty:
        print("Nenhum dado para processar.")
        return

    # Agrupar por algoritmo e calcular a média das métricas de interesse
    metrics_avg = df.groupby('algorithm')[[
        'accuracy', 
        'precisao_ataque', 
        'recall_ataque', 
        'f1_score_ataque',
        'tempo_inferencia'
    ]].mean()

    print("--- Médias Gerais por Algoritmo ---")
    for algorithm, stats in metrics_avg.iterrows():
        print(f"\nAlgoritmo: {algorithm}")
        print(f"  - Acurácia Média:         {stats['accuracy'] * 100:.4f}%")
        print(f"  - Precisão (Ataque) Média: {stats['precisao_ataque'] * 100:.4f}%")
        print(f"  - Recall (Ataque) Média:   {stats['recall_ataque'] * 100:.4f}%")
        print(f"  - F1-Score (Ataque) Média: {stats['f1_score_ataque'] * 100:.4f}%")
        print(f"  - Tempo de Inferência Média: {stats['tempo_inferencia']:.4f} ms")
    print("\n" + "-"*35)

def create_graphs(df):
    """Cria e salva uma série de gráficos para análise dos resultados."""
    if df is None or df.empty:
        print("Nenhum dado para gerar gráficos.")
        return

    output_dir = 'graficos_resultados'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Configurações de estilo dos gráficos
    sns.set_theme(style="whitegrid")

    # 1. Boxplots para comparar a distribuição das métricas
    metrics_to_plot = ['accuracy', 'f1_score_ataque', 'tempo_inferencia']
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='algorithm', y=metric, data=df)
        
        title = f'Distribuição de {metric.replace("_", " ").title()} por Algoritmo'
        if 'accuracy' in metric or 'f1' in metric:
            plt.ylabel(f'{metric.replace("_", " ").title()} (%)')
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}'))
        else:
            plt.ylabel(f'{metric.replace("_", " ").title()} (ms)')

        plt.title(title)
        plt.xlabel("Algoritmo")
        
        save_path = os.path.join(output_dir, f'boxplot_{metric}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Gráfico salvo em: {save_path}")

    # 2. Gráfico de Barras Comparando as Médias
    metrics_avg = df.groupby('algorithm')[metrics_to_plot].mean().reset_index()
    
    # Normalizar os dados para plotar no mesmo gráfico de radar
    plt.figure(figsize=(12, 8))
    metrics_melted = pd.melt(metrics_avg, id_vars="algorithm", var_name="metric", value_name="average")
    
    sns.barplot(x="metric", y="average", hue="algorithm", data=metrics_melted)
    plt.title('Comparativo das Médias das Métricas por Algoritmo')
    plt.ylabel('Valor Médio Normalizado')
    plt.xlabel('Métrica')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'comparativo_medias_barras.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Gráfico salvo em: {save_path}")

    # 3. Scatter Plot: Acurácia vs. Tempo de Inferência
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='accuracy', y='tempo_inferencia', hue='algorithm', alpha=0.7, s=100)
    plt.title('Acurácia vs. Tempo de Inferência')
    plt.xlabel('Acurácia (%)')
    plt.ylabel('Tempo de Inferência (ms)')
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.1f}'))
    plt.legend(title='Algoritmo')
    
    save_path = os.path.join(output_dir, 'scatter_acuracia_vs_tempo.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Gráfico salvo em: {save_path}")

if __name__ == '__main__':
    dataframe = get_data_from_db()
    calculate_and_print_averages(dataframe)
    create_graphs(dataframe)