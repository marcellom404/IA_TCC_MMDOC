# bibliotecas mlxtend e PyFIM tem o Apriori, FP-Growth e Eclat.

import pandas as pd
import numpy as np
import time
import os
import multiprocessing
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Algoritmos de Mineração ---
# mlxtend : Apriori e FP-Growth e se integra bem com pandas.
# mlxtend nao retorna regras em tempo real, nao tem como parar e salvar progresso...
# os testes indicaram que o dataset escolhido e incompatível com mineração
# novo teste, selecionar colunas especificas para testar a mineração.
# funciona mais nao tive resultados uteis apos horas de mineraçao
# fazer o tcc para aprendisado de maquina, e se a mineraçao der resultado acrecentar no tcc.
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
    one_hot_df.to_csv('one_hot_encoded_data.csv', index=False) # para uso no weka


    print("Pré-processamento concluído.")
    return one_hot_df


def _run_apriori(queue, df, min_support, max_len):
    """Função worker para executar o Apriori em um processo separado."""
    print("Iniciando Apriori...")
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True, low_memory=True, verbose=True, max_len=max_len)
    print(f"Encontrados {len(frequent_itemsets)} itemsets frequentes.")

    # Salva os itemsets em um arquivo temporário para evitar problemas com a fila
    temp_file = f"temp_apriori_{os.getpid()}.pkl"
    try:
        frequent_itemsets.to_pickle(temp_file)
        queue.put(temp_file)
    except Exception as e:
        queue.put(e) # Envia a exceção para o processo pai


def _run_fpgrowth(queue, df, min_support):
    """Função worker para executar o FP-Growth em um processo separado."""
    print("Iniciando FP-Growth...")
    frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
    print(f"Encontrados {len(frequent_itemsets)} itemsets frequentes.")

    # Salva os itemsets em um arquivo temporário para evitar problemas com a fila
    temp_file = f"temp_fpgrowth_{os.getpid()}.pkl"
    try:
        frequent_itemsets.to_pickle(temp_file)
        queue.put(temp_file)
    except Exception as e:
        queue.put(e) # Envia a exceção para o processo pai


# def _run_eclat(queue, transactions, min_support):
#     """Função worker para executar o Eclat em um processo separado."""
#     # Eclat da pyfim espera o suporte como uma porcentagem inteira.
#     support_percent = int(min_support * 100)
#     frequent_itemsets_list = eclat(transactions, target='s', supp=support_percent, report='a')
#     queue.put(frequent_itemsets_list)


class BaseMiner:
    """Classe base para compartilhar a lógica de geração e salvamento de regras."""

    def _generate_and_save_rules(self, frequent_itemsets, target_column, max_rules, file_name, min_confidence=0.5, chunk_size=10000):
        print("Gerando e salvando regras de associação...")
        global_caminho = f'{os.getcwd()}/fichas/'

        


        if not os.path.exists(f'{global_caminho}rules'):
            os.makedirs(f'{global_caminho}rules')

        file_name = f'{global_caminho}rules/{file_name}{time.strftime("%Y%m%d_%H%M%S")}.txt'

        if frequent_itemsets.empty:
            print("Nenhum itemset frequente encontrado com o suporte definido. Tente um valor de 'min_support' menor.")
            return

        # Filtra itemsets com suporte zero para evitar erros de divisão por zero.
        frequent_itemsets = frequent_itemsets[frequent_itemsets['support'] > 0]

        if frequent_itemsets.empty:
            print("Nenhum itemset frequente com suporte > 0 foi encontrado. Não é possível gerar regras.")
            return

        print(f"Gerando regras de associação a partir de {len(frequent_itemsets)} itemsets frequentes...")
        # Gera regras com base na métrica de confiança.
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

        if rules.empty:
            print(f"Nenhuma regra encontrada com confiança mínima de {min_confidence}.")
            return

        # Filtra para manter apenas regras que preveem o 'target_column' de forma otimizada.
        # Ex: Queremos regras como {A, B} => {Label_Botnet}
        prefix = f"{target_column}_"
        mask = rules['consequents'].apply(lambda s: any(item.startswith(prefix) for item in s))
        filtered_rules = rules[mask]

        if filtered_rules.empty:
            print(f"Nenhuma regra encontrada que preveja a coluna alvo '{target_column}'.")
            return

        # Ordena as regras pela confiança e lift para mostrar as mais interessantes primeiro.
        # Usar nlargest pode ser mais eficiente que sort_values().head() se max_rules for pequeno.
        print(f"Ordenando e selecionando as {max_rules} melhores regras...")
        if len(filtered_rules) > max_rules:
            top_rules = filtered_rules.nlargest(max_rules, columns=['confidence', 'lift'])
        else:
            top_rules = filtered_rules.sort_values(by=['confidence', 'lift'], ascending=False)

        total_rules_to_save = len(top_rules)

        # Salva as regras em um arquivo de texto de forma otimizada para memória.
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(f"Algoritmo: {self.__class__.__name__}\n")
            f.write(f"Total de regras a serem salvas: {total_rules_to_save}\n")
            f.write("=" * 80 + "\n\n")

            header = True
            for start in range(0, total_rules_to_save, chunk_size):
                end = min(start + chunk_size, total_rules_to_save)
                chunk_df = top_rules.iloc[start:end]

                print(f"Escrevendo lote de {len(chunk_df)} regras (de {start} a {end} de {total_rules_to_save})...")

                # Usar to_csv para streaming dos dados, o que consome menos memória.
                # O separador de tabulação (\t) é usado para manter a legibilidade.
                chunk_df[['antecedents', 'consequents', 'confidence', 'lift']].to_csv(
                    f, index=False, sep='\t', lineterminator='\n', header=header
                )
                header = False # Escreve o cabeçalho apenas para o primeiro lote

        print(f"Sucesso! {total_rules_to_save} regras foram salvas em '{file_name}'.")


class AprioriMiner(BaseMiner):
    def generate_rules(self, df: pd.DataFrame, target_column: str, max_rules: int = 100, timeout_seconds: int = 300, min_support: float = 0.2, max_itemset_len: int = 4, min_confidence: float = 0.5):
        """Gera regras de associação usando o algoritmo Apriori."""
        start_time = time.time()
        print(f"Iniciando Apriori com timeout de {timeout_seconds}s, suporte mínimo de {min_support:.2f} e tamanho máximo de itemset de {max_itemset_len}.")

        one_hot_df = _prepare_data_for_mining(df)


        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=_run_apriori, args=(q, one_hot_df, min_support, max_itemset_len))
        p.start()
        print("Aguardando o processo Apriori...")
        p.join(timeout=timeout_seconds)

        if p.is_alive():
            p.terminate()
            p.join()
            print(f"ERRO: Apriori excedeu o tempo limite de {timeout_seconds} segundos.")
            return

        result = q.get()

        if isinstance(result, Exception):
            print(f"ERRO no processo Apriori: {result}")
            return

        temp_file = result
        try:
            frequent_itemsets = pd.read_pickle(temp_file)
            print(f"Encontrados {len(frequent_itemsets)} itemsets frequentes com suporte >= {min_support:.2f}.")
            self._generate_and_save_rules(frequent_itemsets, target_column, max_rules, "apriori_rules.txt", min_confidence=min_confidence)
        except Exception as e:
            print(f"ERRO ao processar o resultado do Apriori: {e}")
        finally:
            # Garante que o arquivo temporário seja removido apenas se for um caminho de arquivo
            if isinstance(temp_file, str) and os.path.exists(temp_file):
                os.remove(temp_file)

        print(f"Apriori concluído em {time.time() - start_time:.2f} segundos.")


class FPGrowthMiner(BaseMiner):
    def generate_rules(self, df: pd.DataFrame, target_column: str, max_rules: int = 100, timeout_seconds: int = 300, min_support: float = 0.2, min_confidence: float = 0.5):
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

        result = q.get()

        if isinstance(result, Exception):
            print(f"ERRO no processo FP-Growth: {result}")
            return

        temp_file = result
        try:
            frequent_itemsets = pd.read_pickle(temp_file)
            self._generate_and_save_rules(frequent_itemsets, target_column, max_rules, "fpgrowth_rules.txt", min_confidence=min_confidence)
        except Exception as e:
            print(f"ERRO ao processar o resultado do FP-Growth: {e}")
        finally:
            # Garante que o arquivo temporário seja removido apenas se for um caminho de arquivo
            if isinstance(temp_file, str) and os.path.exists(temp_file):
                os.remove(temp_file)

        print(f"FP-Growth concluído em {time.time() - start_time:.2f} segundos.")



    # Como o tamanho do dataset e enorme vou criar datasets com colunas aleatórias diminuindo o uso de memória.

def create_random_datasets(
    original_df: pd.DataFrame,
    target_column: str,
    n_columns_per_dataset: int,
    n_datasets: int
) -> list[pd.DataFrame]:
    """
    Cria uma lista de DataFrames com colunas aleatórias do DataFrame original,
    mantendo uma coluna-alvo sempre presente.

    Args:
        original_df (pd.DataFrame): O DataFrame original.
        target_column (str): O nome da coluna que deve estar presente em todos
                              os DataFrames de saída.
        n_columns_per_dataset (int): O número de colunas em cada DataFrame de saída.
                                     Deve ser no mínimo 2 (a coluna alvo + 1).
        n_datasets (int): O número de DataFrames a serem criados.

    Returns:
        list[pd.DataFrame]: Uma lista de DataFrames criados.
    """
    # Verificação de erros
    if target_column not in original_df.columns:
        raise ValueError(f"A coluna-alvo '{target_column}' não existe no DataFrame original.")

    if n_columns_per_dataset < 2:
        raise ValueError("O número de colunas por dataset deve ser no mínimo 2.")

    # Lista de colunas disponíveis, excluindo a coluna-alvo
    available_columns = [col for col in original_df.columns if col != target_column]

    # Número de colunas a serem selecionadas aleatoriamente
    n_random_columns = n_columns_per_dataset - 1

    # Verificação se há colunas suficientes para a seleção
    if len(available_columns) < n_random_columns:
        raise ValueError(
            f"O número de colunas disponíveis ({len(available_columns)}) "
            f"é insuficiente para criar datasets com {n_columns_per_dataset} colunas."
        )

    # Lista para armazenar os novos datasets
    list_of_datasets = []

    for _ in range(n_datasets):
        # Seleciona aleatoriamente as colunas, sem repetição
        random_columns = np.random.choice(
            available_columns,
            size=n_random_columns,
            replace=False
        ).tolist()

        # Adiciona a coluna-alvo de volta
        selected_columns = [target_column] + random_columns

        # Cria o novo DataFrame e adiciona à lista
        new_df = original_df[selected_columns].copy()
        list_of_datasets.append(new_df)

    return list_of_datasets

def create_non_repeating_datasets(
    original_df: pd.DataFrame,
    target_column: str,
    n_columns_per_dataset: int
) -> list[pd.DataFrame]:
    """
    Cria uma lista de DataFrames com colunas aleatórias do DataFrame original,
    mantendo uma coluna-alvo e garantindo que cada coluna aleatória seja usada
    apenas uma vez em toda a lista de datasets. O processo continua até que não
    haja colunas suficientes para criar um novo dataset.

    Args:
        original_df (pd.DataFrame): O DataFrame original.
        target_column (str): O nome da coluna que deve estar presente em todos
                              os DataFrames de saída.
        n_columns_per_dataset (int): O número de colunas em cada DataFrame de saída.
                                     Deve ser no mínimo 2 (a coluna alvo + 1).

    Returns:
        list[pd.DataFrame]: Uma lista de DataFrames criados.
    """
    # Verificação de erros
    if target_column not in original_df.columns:
        raise ValueError(f"A coluna-alvo '{target_column}' não existe no DataFrame original.")

    if n_columns_per_dataset < 2:
        raise ValueError("O número de colunas por dataset deve ser no mínimo 2.")

    # Lista de colunas disponíveis, excluindo a coluna-alvo
    available_columns = [col for col in original_df.columns if col != target_column]
    n_random_columns_to_select = n_columns_per_dataset - 1

    # Verificação inicial se há colunas suficientes para pelo menos um dataset
    if len(available_columns) < n_random_columns_to_select:
        print(
            f"Aviso: Não há colunas suficientes para criar datasets com "
            f"{n_columns_per_dataset} colunas (necessário no mínimo "
            f"{n_random_columns_to_select + 1} colunas além da coluna-alvo)."
        )
        return []

    # Embaralha as colunas disponíveis para garantir a aleatoriedade
    np.random.shuffle(available_columns)

    list_of_datasets = []
    start_index = 0

    while start_index + n_random_columns_to_select <= len(available_columns):
        # Seleciona as colunas para o dataset atual
        selected_random_columns = available_columns[
            start_index : start_index + n_random_columns_to_select
        ]

        # Constrói a lista final de colunas, incluindo a coluna-alvo
        selected_columns = [target_column] + selected_random_columns

        # Cria o novo DataFrame
        new_df = original_df[selected_columns].copy()
        list_of_datasets.append(new_df)

        # Atualiza o índice para a próxima seleção
        start_index += n_random_columns_to_select

    return list_of_datasets

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
