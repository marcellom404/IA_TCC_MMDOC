
# bibliotecas mlxtend e PyFIM tem o Apriori, FP-Growth e Eclat.

import pandas as pd
import numpy as np
import time
import os
import multiprocessing

# --- Algoritmos de Mineração ---
# mlxtend é ótimo para Apriori e FP-Growth e se integra bem com pandas.
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
# PyFIM é uma implementação eficiente para Eclat.
# ppyFIM ta com um erro de limite de caminho.
# from fim import eclat





def _prepare_data_for_mining(df: pd.DataFrame, n_bins: int = 4) -> pd.DataFrame:
    """
    Prepara o DataFrame para mineração de regras de associação.
    1. Discretiza colunas numéricas em 'n_bins' (quartis).
    2. Converte todas as colunas para o formato one-hot encoded.
    """
    print("Iniciando pré-processamento dos dados...")
    processed_df = df.copy()

    # Remove colunas de identificadores que não são úteis para encontrar padrões.
    cols_to_drop = ["id", "Timestamp", "Src IP", "Dst IP", "Flow ID"]
    for col in cols_to_drop:
        if col in processed_df.columns:
            processed_df.drop(columns=col, inplace=True)

    # Discretiza colunas numéricas. Algoritmos de associação funcionam com itens categóricos.
    numerical_cols = processed_df.select_dtypes(include=np.number).columns
    for col in numerical_cols:
        try:
            # pd.qcut cria bins com base em quantis (ex: 0-25%, 25-50%, etc.)
            # Isso lida melhor com distribuições de dados desiguais.
            processed_df[col] = pd.qcut(processed_df[col], q=n_bins, labels=False, duplicates='drop')
            # Adiciona o nome da coluna original para evitar colisões de nomes após o one-hot encoding.
            processed_df[col] = f'{col}_bin_' + processed_df[col].astype(str)
        except ValueError:
            # Se qcut falhar (poucos valores únicos), apenas converte para string.
            print(f"Aviso: Não foi possível discretizar a coluna '{col}'. Convertendo para string.")
            processed_df[col] = processed_df[col].astype(str)

    print("Convertendo dados para formato one-hot encoded...")
    # pd.get_dummies cria uma coluna para cada valor único, com True/False.
    # Este é o formato que os algoritmos esperam.
    one_hot_df = pd.get_dummies(processed_df.astype(str))
    print("Pré-processamento concluído.")
    return one_hot_df


def _run_apriori(queue, df, min_support):
    """Função worker para executar o Apriori em um processo separado."""
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    queue.put(frequent_itemsets)


def _run_fpgrowth(queue, df, min_support):
    """Função worker para executar o FP-Growth em um processo separado."""
    frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
    queue.put(frequent_itemsets)


# def _run_eclat(queue, transactions, min_support):
#     """Função worker para executar o Eclat em um processo separado."""
#     # Eclat da pyfim espera o suporte como uma porcentagem inteira.
#     support_percent = int(min_support * 100)
#     frequent_itemsets_list = eclat(transactions, target='s', supp=support_percent, report='a')
#     queue.put(frequent_itemsets_list)


class BaseMiner:
    """Classe base para compartilhar a lógica de geração e salvamento de regras."""

    def _generate_and_save_rules(self, frequent_itemsets, target_column, max_rules, file_name, min_confidence=0.5):
        if frequent_itemsets.empty:
            print("Nenhum itemset frequente encontrado com o suporte definido. Tente um valor de 'min_support' menor.")
            return

        print(f"Gerando regras de associação a partir de {len(frequent_itemsets)} itemsets frequentes...")
        # Gera regras com base na métrica de confiança.
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

        if rules.empty:
            print(f"Nenhuma regra encontrada com confiança mínima de {min_confidence}.")
            return

        # Filtra para manter apenas regras que preveem o 'target_column'.
        # Ex: Queremos regras como {A, B} => {Label_Botnet}
        rules['consequents_str'] = rules['consequents'].apply(lambda x: ','.join(list(x)))
        filtered_rules = rules[rules['consequents_str'].str.startswith(f"{target_column}_")]

        if filtered_rules.empty:
            print(f"Nenhuma regra encontrada que preveja a coluna alvo '{target_column}'.")
            return

        # Ordena as regras pela confiança e lift para mostrar as mais interessantes primeiro.
        sorted_rules = filtered_rules.sort_values(by=['confidence', 'lift'], ascending=False).head(max_rules)

        # Salva as regras em um arquivo de texto.
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(f"Algoritmo: {self.__class__.__name__}\n")
            f.write(f"Total de regras geradas (filtradas e ordenadas): {len(sorted_rules)}\n")
            f.write("=" * 80 + "\n\n")
            f.write(sorted_rules[['antecedents', 'consequents', 'confidence', 'lift']].to_string())

        print(f"Sucesso! {len(sorted_rules)} regras foram salvas em '{file_name}'.")


class AprioriMiner(BaseMiner):
    def generate_rules(self, df: pd.DataFrame, target_column: str, max_rules: int = 100, timeout_seconds: int = 300, min_support: float = 0.2):
        """Gera regras de associação usando o algoritmo Apriori."""
        start_time = time.time()
        print(f"Iniciando Apriori com timeout de {timeout_seconds}s e suporte mínimo de {min_support:.2f}.")

        one_hot_df = _prepare_data_for_mining(df)

        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=_run_apriori, args=(q, one_hot_df, min_support))
        p.start()
        p.join(timeout=timeout_seconds)

        if p.is_alive():
            p.terminate()
            p.join()
            print(f"ERRO: Apriori excedeu o tempo limite de {timeout_seconds} segundos.")
            return

        frequent_itemsets = q.get()
        self._generate_and_save_rules(frequent_itemsets, target_column, max_rules, "apriori_rules.txt")
        print(f"Apriori concluído em {time.time() - start_time:.2f} segundos.")


class FPGrowthMiner(BaseMiner):
    def generate_rules(self, df: pd.DataFrame, target_column: str, max_rules: int = 100, timeout_seconds: int = 300, min_support: float = 0.2):
        """Gera regras de associação usando o algoritmo FP-Growth."""
        start_time = time.time()
        print(f"Iniciando FP-Growth com timeout de {timeout_seconds}s e suporte mínimo de {min_support:.2f}.")

        one_hot_df = _prepare_data_for_mining(df)

        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=_run_fpgrowth, args=(q, one_hot_df, min_support))
        p.start()
        p.join(timeout=timeout_seconds)

        if p.is_alive():
            p.terminate()
            p.join()
            print(f"ERRO: FP-Growth excedeu o tempo limite de {timeout_seconds} segundos.")
            return

        frequent_itemsets = q.get()
        self._generate_and_save_rules(frequent_itemsets, target_column, max_rules, "fpgrowth_rules.txt")
        print(f"FP-Growth concluído em {time.time() - start_time:.2f} segundos.")

# por algum motivo o windows não suporta, então Eclat está comentado.  
# class EclatMiner(BaseMiner):
#     def generate_rules(self, df: pd.DataFrame, target_column: str, max_rules: int = 100, timeout_seconds: int = 300, min_support: float = 0.2):
#         """Gera regras de associação usando o algoritmo Eclat."""
#         start_time = time.time()
#         print(f"Iniciando Eclat com timeout de {timeout_seconds}s e suporte mínimo de {min_support:.2f}.")

#         one_hot_df = _prepare_data_for_mining(df)

#         # Eclat da pyfim espera uma lista de transações (lista de listas de itens).
#         print("Convertendo DataFrame para formato de transação...")
#         # A conversão linha por linha é lenta. Uma abordagem vetorizada é muito mais rápida.
#         # Obtém a matriz numpy subjacente.
#         arr = one_hot_df.to_numpy()
#         # Usa a indexação booleana para obter os nomes das colunas para cada linha onde o valor é True.
#         transactions = [one_hot_df.columns[row].tolist() for row in arr]

#         q = multiprocessing.Queue()
#         p = multiprocessing.Process(target=_run_eclat, args=(q, transactions, min_support))
#         p.start()
#         p.join(timeout=timeout_seconds)

#         if p.is_alive():
#             p.terminate()
#             p.join()
#             print(f"ERRO: Eclat excedeu o tempo limite de {timeout_seconds} segundos.")
#             return

#         # Converte a saída do Eclat (lista de tuplas) para o formato de DataFrame que o mlxtend espera.
#         eclat_result = q.get()
#         if not eclat_result:
#             print("Nenhum itemset frequente encontrado pelo Eclat.")
#             return

#         # A saída do Eclat com report='a' é uma lista de tuplas: (itemset, suporte_absoluto)
#         # O itemset é uma tupla de strings. O suporte é a contagem de transações.
#         # Precisamos converter para o formato do mlxtend:
#         # - support: suporte relativo (float)
#         # - itemsets: frozenset
#         total_transactions = len(transactions)
#         frequent_itemsets = pd.DataFrame(
#             [
#                 (item[1] / total_transactions, frozenset(item[0]))
#                 for item in eclat_result
#             ],
#             columns=['support', 'itemsets']
#         )

#         if frequent_itemsets.empty:
#             print("Nenhum itemset frequente encontrado pelo Eclat após a conversão.")
#             return

#         self._generate_and_save_rules(frequent_itemsets, target_column, max_rules, "eclat_rules.txt")
#         print(f"Eclat concluído em {time.time() - start_time:.2f} segundos.")