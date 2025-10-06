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
    try:
        df = pd.read_sql_query("SELECT * FROM results", conn)
    except pd.io.sql.DatabaseError:
        print("A tabela 'results' parece estar vazia ou com erro.")
        return None
    finally:
        conn.close()
        
    if 'parameters' in df.columns:
        parameters_df = df['parameters'].apply(lambda x: json.loads(x)).apply(pd.Series)
        df = pd.concat([df.drop('parameters', axis=1), parameters_df], axis=1)
        
    return df

def calculate_and_print_averages(df):
    """Calcula e exibe as médias das principais métricas para cada algoritmo."""
    if df is None or df.empty:
        print("Nenhum dado para processar.")
        return

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

    sns.set_theme(style="whitegrid")

    metric_translation = {
        'accuracy': 'Acurácia',
        'f1_score_ataque': 'F1-Score (Ataque)',
        'tempo_inferencia': 'Tempo de Inferência'
    }

    metrics_to_plot = ['accuracy', 'f1_score_ataque', 'tempo_inferencia']
    for metric in metrics_to_plot:
        metric_pt = metric_translation.get(metric, metric.replace(" ", " ").title())
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='algorithm', y=metric, data=df)
        
        title = f'Distribuição de {metric_pt} por Algoritmo'
        if 'accuracy' in metric or 'f1' in metric:
            plt.ylabel(f'{metric_pt} (%)')
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}'))
        else:
            plt.ylabel(f'{metric_pt} (ms)')

        plt.title(title)
        plt.xlabel("Algoritmo")
        
        save_path = os.path.join(output_dir, f'boxplot_{metric}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Gráfico salvo em: {save_path}")

    metrics_avg = df.groupby('algorithm')[metrics_to_plot].mean().reset_index()
    
    # Renomeia as colunas de métricas para o gráfico de barras
    metrics_avg_renamed = metrics_avg.rename(columns=metric_translation)
    
    plt.figure(figsize=(12, 8))
    metrics_melted = pd.melt(metrics_avg_renamed, id_vars="algorithm", var_name="Métrica", value_name="average")
    
    sns.barplot(x="Métrica", y="average", hue="algorithm", data=metrics_melted)
    plt.title('Comparativo das Médias das Métricas por Algoritmo')
    plt.ylabel('Valor Médio')
    plt.xlabel('Métrica')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'comparativo_medias_barras.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Gráfico salvo em: {save_path}")

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

def create_parameter_graphs(df):
    """Cria gráficos que relacionam os parâmetros de treinamento com as métricas."""
    if df is None or df.empty:
        print("Nenhum dado para gerar gráficos de parâmetros.")
        return

    output_dir = 'graficos_resultados/parametros'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    sns.set_theme(style="whitegrid")

    translation = {
        'accuracy': 'Acurácia',
        'f1_score_ataque': 'F1-Score (Ataque)',
        'tempo_inferencia': 'Tempo de Inferência',
        'n_neighbors': 'Número de Vizinhos (K)',
        'hidden_layer_sizes': 'Tamanho da Camada Oculta',
        'n_estimators': 'Número de Estimadores'
    }

    algorithms_params = {
        'K-NN': 'n_neighbors',
        'MLP': 'hidden_layer_sizes',
        'Random Forest': 'n_estimators'
    }
    
    metrics_to_plot = ['accuracy', 'f1_score_ataque', 'tempo_inferencia']

    for alg, param_name in algorithms_params.items():
        alg_df = df[df['algorithm'] == alg].dropna(subset=[param_name])
        if alg_df.empty:
            print(f"Sem dados para o algoritmo {alg} com o parâmetro {param_name}.")
            continue
        
        # Garante que a coluna de parâmetro possa ser ordenada corretamente
        if param_name == 'hidden_layer_sizes':
            # Extrai o primeiro número da tupla de tamanhos de camada para ordenação
            try:
                # Trata tanto strings de tuplas quanto números
                alg_df.loc[:, param_name] = alg_df[param_name].apply(lambda x: int(eval(x)[0]) if isinstance(x, str) else int(x))
            except (SyntaxError, TypeError, IndexError, ValueError):
                print(f"Aviso: Não foi possível processar '{param_name}' para o algoritmo {alg}. Pulando gráfico.")
                continue
        
        for metric in metrics_to_plot:
            plt.figure(figsize=(12, 7))
            
            try:
                param_grouped = alg_df.groupby(param_name)[metric].mean().reset_index()
            except Exception as e:
                print(f"Erro ao agrupar dados para {alg} com parâmetro {param_name}: {e}")
                continue

            sns.lineplot(data=param_grouped, x=param_name, y=metric, marker='o')
            
            metric_pt = translation.get(metric, metric.replace(" ", " ").title())
            param_name_pt = translation.get(param_name, param_name.replace(" ", " ").title())
            
            title = f'{metric_pt} vs. {param_name_pt} para {alg}'
            plt.title(title)
            plt.xlabel(param_name_pt)
            
            if 'accuracy' in metric or 'f1' in metric:
                plt.ylabel(f'{metric_pt} (%)')
                plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.2f}'))
            else:
                plt.ylabel(f'{metric_pt} (ms)')

            plt.tight_layout()
            save_path = os.path.join(output_dir, f'{alg.replace(" ", "_")}_{metric}_vs_{param_name}.png')
            plt.savefig(save_path)
            plt.close()
            print(f"Gráfico de parâmetro salvo em: {save_path}")


if __name__ == '__main__':
    dataframe = get_data_from_db()
    if dataframe is not None:
        calculate_and_print_averages(dataframe)
        # create_graphs(dataframe)
        # create_parameter_graphs(dataframe)
    else:
        print("Processamento encerrado pois não há dados.")