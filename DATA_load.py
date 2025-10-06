import os
import manipul_arquivo as mov
import funcoes_teste_dataset as teste
import random
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
import numpy as np



global alvo 
alvo = "Label"

if os.path.exists("datasets"):

    path = "datasets/1"
else:
    path = kagglehub.dataset_download("ernie55ernie/improved-cicids2017-and-csecicids2018")
    #move o dataset para fora dos arquivos temporarios, salvando em datasets/
    mov.save_dataset_path_to_db(mov.move_dataset(path))
# informaçoes sobre os datasets no info_dataset.py 'CSECICIDS2018_improved/Friday-02-03-2018.csv'
DATASET_NAME ="CSECICIDS2018_improved\merged_dataset.csv"
print(path + "/" + DATASET_NAME)
DATASET_PATH = path + "/" + DATASET_NAME
# --- CONFIGURAÇÃO DA AMOSTRAGEM ---
# Defina a porcentagem de ataques desejada no corte do dataset (ex: 0.2 para 20%)
# Se não houver ataques suficientes, o restante será preenchido com amostras benignas.
porcentagem_ataque_desejada = 0.15
num_linhas_desejado = 20000

print("Analisando a distribuição de classes no arquivo completo (pode levar um momento)...")
try:
    # Lê apenas a coluna 'Label' para identificar os índices de cada classe sem estourar a memória
    label_df = pd.read_csv(DATASET_PATH, usecols=['Label'], dtype={'Label': 'category'})
    attack_indices = label_df.index[label_df['Label'] != 'BENIGN'].tolist()
    benign_indices = label_df.index[label_df['Label'] == 'BENIGN'].tolist()

    num_ataques_desejado = int(num_linhas_desejado * porcentagem_ataque_desejada)

    # --- NOVA LÓGICA PARA AMOSTRAGEM DE ATAQUES ---
    # Identifica os tipos de ataque únicos (excluindo 'BENIGN')
    attack_labels = label_df[label_df['Label'] != 'BENIGN']['Label'].unique()
    num_attack_types = len(attack_labels)
    ataques_a_ler_indices = []

    if num_attack_types > 0:
        print(f"Encontrados {num_attack_types} tipos de ataque. Distribuindo {num_ataques_desejado} amostras de ataque entre eles.")
        
        base_samples_per_attack = num_ataques_desejado // num_attack_types
        remainder = num_ataques_desejado % num_attack_types

        for i, attack_label in enumerate(attack_labels):
            num_samples_for_this_attack = base_samples_per_attack + (1 if i < remainder else 0)

            current_attack_indices = label_df.index[label_df['Label'] == attack_label].tolist()

            if len(current_attack_indices) < num_samples_for_this_attack:
                print(f"AVISO: Para o ataque '{attack_label}', existem apenas {len(current_attack_indices)} amostras, "
                      f"menos que as {num_samples_for_this_attack} desejadas. Usando todas as disponíveis.")
                samples_to_take = len(current_attack_indices)
            else:
                samples_to_take = num_samples_for_this_attack
            
            if samples_to_take > 0:
                ataques_a_ler_indices.extend(random.sample(current_attack_indices, samples_to_take))
    
    if not ataques_a_ler_indices:
        print("AVISO: Nenhum ataque encontrado no dataset para amostragem.")
    else:
        print(f"Total de {len(ataques_a_ler_indices)} amostras de ataque selecionadas de {num_attack_types} tipos.")

    # Preenche o restante com benignos
    num_benignos_a_ler = num_linhas_desejado - len(ataques_a_ler_indices)
    benignos_a_ler_indices = random.sample(benign_indices, num_benignos_a_ler)

    indices_a_manter = set(ataques_a_ler_indices + benignos_a_ler_indices)

    # Lê o CSV pulando as linhas que não queremos.
    print("Lendo amostra estratificada do disco...")
    df = pd.read_csv(DATASET_PATH, skiprows=lambda i: i > 0 and (i - 1) not in indices_a_manter)
    print(f"Amostra de {len(df)} linhas carregada, com {len(ataques_a_ler_indices)} ataques.")

except Exception as e:
    print(f"FALHA NA AMOSTRAGEM ESTRATIFICADA: {e}.")
    print("Usando amostragem aleatória simples como fallback.")
    num_linhas_total = sum(1 for row in open(DATASET_PATH)) -1
    skip_indices = sorted(random.sample(range(1, num_linhas_total + 1),
                                         num_linhas_total - num_linhas_desejado))
    df = pd.read_csv(DATASET_PATH, skiprows=skip_indices)

# colunas que precisam ser removidas
columns_to_drop = ["id", "Attempted Category", "Timestamp", "Src IP", "Dst IP","Flow ID","Fwd URG Flags",'Bwd URG Flags', 'URG Flag Count']
df.drop(columns=columns_to_drop, inplace=True)

# Substitui valores infinitos por NaN e remove as linhas correspondentes
df.replace([np.inf, -np.inf], np.nan, inplace=True)
linhas_com_nan = df.isnull().any(axis=1).sum()
if linhas_com_nan > 0:
    print(f"Limpando {linhas_com_nan} linhas que continham valores infinitos.")
    df.dropna(inplace=True)

# Medida de segurança para remover colunas que vazam a resposta (ex: Label_BENIGN)
leaky_cols = [col for col in df.columns if col.startswith('Label_')]
if leaky_cols:
    print(f"ALERTA DE SEGURANÇA: Encontradas e removidas colunas com vazamento de dados: {leaky_cols}")
    df.drop(columns=leaky_cols, inplace=True)

print("\n--- Distribuição de Classes no DataFrame Amostrado ---")
print(df[alvo].value_counts())

# --- FILTRAGEM DE CLASSES RARAS ---
print("\n--- Filtrando classes com menos de 100 amostras ---")
class_counts = df[alvo].value_counts()
rare_classes = class_counts[class_counts < 100].index.tolist()
# Ensure 'BENIGN' is not considered a rare class to be merged
if 'BENIGN' in rare_classes:
    rare_classes.remove('BENIGN')

# Find a suitable class to merge the rare ones into (most common attack class)
ataques_counts = class_counts[class_counts.index != 'BENIGN']
if not ataques_counts.empty:
    # Select the most frequent attack class that is not in the rare_classes list
    valid_ataques_counts = ataques_counts.drop(rare_classes, errors='ignore')
    if not valid_ataques_counts.empty:
        merge_target_class = valid_ataques_counts.idxmax()

        if rare_classes:
            print(f"Classes raras a serem mescladas: {rare_classes}")
            print(f"Classe alvo para mesclagem: '{merge_target_class}'")
            df[alvo] = df[alvo].replace(rare_classes, merge_target_class)
            print("\n--- Distribuição de Classes Após a Mesclagem ---")
            print(df[alvo].value_counts())
        else:
            print("Nenhuma classe de ataque rara (com menos de 100 amostras) para mesclar.")
    else:
        print("Nenhuma classe de ataque com 100 ou mais amostras encontrada para servir como alvo de mesclagem.")
else:
    print("Nenhuma classe de ataque encontrada para avaliar a raridade.")

# "label" e a coluna que indica o tipo de fluxo de rede, sendo BENIGN o trafego comum, outros sao algum tipo de tentativa de intrusão
# print(df["Label"].value_counts())

def get_dados_amostra():
    tentativa = 0
    while True:
        tentativa += 1
        print(f"\n--- Tentativa de divisão de dados nº {tentativa} ---")
        # ---Filtrar o dataframe principal ---
        class_counts1 = df[alvo].value_counts()
        rare_classes1 = class_counts1[class_counts1 < 2].index
        if not rare_classes1.empty:
            df_filtered = df[~df[alvo].isin(rare_classes1)]
        else:
            df_filtered = df

        # --- divisões ---
        treinamento, df_restante = train_test_split(
            df_filtered,
            test_size=0.5,
            stratify=df_filtered[alvo],
            random_state=random.randint(1, 1000)
        )

       
        class_counts2 = df_restante[alvo].value_counts()
        rare_classes2 = class_counts2[class_counts2 < 2].index
        if not rare_classes2.empty:
            df_restante_filtered = df_restante[~df_restante[alvo].isin(rare_classes2)]
        else:
            df_restante_filtered = df_restante

        
        if df_restante_filtered.empty or len(df_restante_filtered) < 2:
            print("Não foi possível criar conjuntos de validação/teste. Repetindo...")
            continue

        test_size = 0.33
        if len(df_restante_filtered) * test_size < 1:
            test_size = 1.0

        validacao, teste = train_test_split(
            df_restante_filtered,
            test_size=test_size,
            stratify=df_restante_filtered[alvo],
            random_state=random.randint(1, 1000)
        )

        
        labels_treino = set(treinamento[alvo].unique())
        labels_validacao = set(validacao[alvo].unique())
        labels_teste = set(teste[alvo].unique())

        validacao_ok = labels_validacao.issubset(labels_treino)
        teste_ok = labels_teste.issubset(labels_treino)

        if not validacao.empty and not teste.empty and validacao_ok and teste_ok:
            
            break
        else:
            
            if not validacao_ok:
                print(f"  - Classes na validação não contidas no treino: {labels_validacao - labels_treino}")
            if not teste_ok:
                print(f"  - Classes no teste não contidas no treino: {labels_teste - labels_treino}")
                


    return treinamento, validacao, teste
if __name__ == '__main__':
    treinamento, validacao, teste = get_dados_amostra()
    print(f"\n--- Análise Final do DataFrame ---")
    print(f"Total de {len(df)} linhas no conjunto original")
    print(f"Treinamento: {len(treinamento)} linhas")
    print(f"Validação: {len(validacao)} linhas")
    print(f"Teste: {len(teste)} linhas")


# print("\nDistribuição de Classes (Treinamento):")
# print(treinamento[alvo].value_counts())
# print("\nDistribuição de Classes (Validação):")
# print(validacao[alvo].value_counts())
# print("\nDistribuição de Classes (Teste):")
# print(teste[alvo].value_counts())

# variaveis apos pre-processamento

# ['Src Port_Src Port_bin_0' 'Src Port_Src Port_bin_1'
#  'Src Port_Src Port_bin_2' 'Src Port_Src Port_bin_3'
#  'Dst Port_Dst Port_bin_0' 'Dst Port_Dst Port_bin_1'
#  'Protocol_Protocol_bin_0' 'Protocol_Protocol_bin_1'
#  'Flow Duration_Flow Duration_bin_0' 'Flow Duration_Flow Duration_bin_1'
#  'Flow Duration_Flow Duration_bin_2' 'Flow Duration_Flow Duration_bin_3'
#  'Total Fwd Packet_Total Fwd Packet_bin_0'
#  'Total Fwd Packet_Total Fwd Packet_bin_1'
#  'Total Bwd packets_Total Bwd packets_bin_0'
#  'Total Bwd packets_Total Bwd packets_bin_1'
#  'Total Length of Fwd Packet_Total Length of Fwd Packet_bin_0'
#  'Total Length of Fwd Packet_Total Length of Fwd Packet_bin_1'
#  'Total Length of Fwd Packet_Total Length of Fwd Packet_bin_2'
#  'Total Length of Fwd Packet_Total Length of Fwd Packet_bin_3'
#  'Total Length of Bwd Packet_Total Length of Bwd Packet_bin_0'
#  'Total Length of Bwd Packet_Total Length of Bwd Packet_bin_1'
#  'Total Length of Bwd Packet_Total Length of Bwd Packet_bin_2'
#  'Total Length of Bwd Packet_Total Length of Bwd Packet_bin_3'
#  'Fwd Packet Length Max_Fwd Packet Length Max_bin_0'
#  'Fwd Packet Length Max_Fwd Packet Length Max_bin_1'
#  'Fwd Packet Length Max_Fwd Packet Length Max_bin_2'
#  'Fwd Packet Length Max_Fwd Packet Length Max_bin_3'
#  'Fwd Packet Length Min_Fwd Packet Length Min_bin_0'
#  'Fwd Packet Length Min_Fwd Packet Length Min_bin_1'
#  'Fwd Packet Length Min_Fwd Packet Length Min_bin_2'
#  'Fwd Packet Length Min_Fwd Packet Length Min_bin_3'
#  'Fwd Packet Length Mean_Fwd Packet Length Mean_bin_0'
#  'Fwd Packet Length Mean_Fwd Packet Length Mean_bin_1'
#  'Fwd Packet Length Mean_Fwd Packet Length Mean_bin_2'
#  'Fwd Packet Length Mean_Fwd Packet Length Mean_bin_3'
#  'Fwd Packet Length Std_Fwd Packet Length Std_bin_nan'
#  'Bwd Packet Length Max_Bwd Packet Length Max_bin_0'
#  'Bwd Packet Length Max_Bwd Packet Length Max_bin_1'
#  'Bwd Packet Length Max_Bwd Packet Length Max_bin_2'
#  'Bwd Packet Length Max_Bwd Packet Length Max_bin_3'
#  'Bwd Packet Length Min_Bwd Packet Length Min_bin_0'
#  'Bwd Packet Length Min_Bwd Packet Length Min_bin_1'
#  'Bwd Packet Length Min_Bwd Packet Length Min_bin_2'
#  'Bwd Packet Length Min_Bwd Packet Length Min_bin_3'
#  'Bwd Packet Length Mean_Bwd Packet Length Mean_bin_0'
#  'Bwd Packet Length Mean_Bwd Packet Length Mean_bin_1'
#  'Bwd Packet Length Mean_Bwd Packet Length Mean_bin_2'
#  'Bwd Packet Length Mean_Bwd Packet Length Mean_bin_3'
#  'Bwd Packet Length Std_Bwd Packet Length Std_bin_nan'
#  'Flow Bytes/s_Flow Bytes/s_bin_0' 'Flow Bytes/s_Flow Bytes/s_bin_1'
#  'Flow Bytes/s_Flow Bytes/s_bin_2' 'Flow Bytes/s_Flow Bytes/s_bin_3'
#  'Flow Packets/s_Flow Packets/s_bin_0'
#  'Flow Packets/s_Flow Packets/s_bin_1'
#  'Flow Packets/s_Flow Packets/s_bin_2'
#  'Flow Packets/s_Flow Packets/s_bin_3' 'Flow IAT Mean_Flow IAT Mean_bin_0'
#  'Flow IAT Mean_Flow IAT Mean_bin_1' 'Flow IAT Mean_Flow IAT Mean_bin_2'
#  'Flow IAT Mean_Flow IAT Mean_bin_3' 'Flow IAT Std_Flow IAT Std_bin_0'
#  'Flow IAT Std_Flow IAT Std_bin_1' 'Flow IAT Std_Flow IAT Std_bin_2'
#  'Flow IAT Max_Flow IAT Max_bin_0' 'Flow IAT Max_Flow IAT Max_bin_1'
#  'Flow IAT Max_Flow IAT Max_bin_2' 'Flow IAT Max_Flow IAT Max_bin_3'
#  'Flow IAT Min_Flow IAT Min_bin_0' 'Flow IAT Min_Flow IAT Min_bin_1'
#  'Flow IAT Min_Flow IAT Min_bin_2' 'Flow IAT Min_Flow IAT Min_bin_3'
#  'Fwd IAT Total_Fwd IAT Total_bin_0' 'Fwd IAT Total_Fwd IAT Total_bin_1'
#  'Fwd IAT Total_Fwd IAT Total_bin_2' 'Fwd IAT Mean_Fwd IAT Mean_bin_0'
#  'Fwd IAT Mean_Fwd IAT Mean_bin_1' 'Fwd IAT Mean_Fwd IAT Mean_bin_2'
#  'Fwd IAT Std_Fwd IAT Std_bin_nan' 'Fwd IAT Max_Fwd IAT Max_bin_0'
#  'Fwd IAT Max_Fwd IAT Max_bin_1' 'Fwd IAT Max_Fwd IAT Max_bin_2'
#  'Fwd IAT Min_Fwd IAT Min_bin_0' 'Fwd IAT Min_Fwd IAT Min_bin_1'
#  'Fwd IAT Min_Fwd IAT Min_bin_2' 'Bwd IAT Total_Bwd IAT Total_bin_0'
#  'Bwd IAT Total_Bwd IAT Total_bin_1' 'Bwd IAT Total_Bwd IAT Total_bin_2'
#  'Bwd IAT Mean_Bwd IAT Mean_bin_0' 'Bwd IAT Mean_Bwd IAT Mean_bin_1'
#  'Bwd IAT Mean_Bwd IAT Mean_bin_2' 'Bwd IAT Std_Bwd IAT Std_bin_nan'
#  'Bwd IAT Max_Bwd IAT Max_bin_0' 'Bwd IAT Max_Bwd IAT Max_bin_1'
#  'Bwd IAT Max_Bwd IAT Max_bin_2' 'Bwd IAT Min_Bwd IAT Min_bin_0'
#  'Bwd IAT Min_Bwd IAT Min_bin_1' 'Bwd IAT Min_Bwd IAT Min_bin_2'
#  'Fwd PSH Flags_Fwd PSH Flags_bin_nan'
#  'Bwd PSH Flags_Bwd PSH Flags_bin_nan'
#  'Fwd RST Flags_Fwd RST Flags_bin_nan' 'Bwd RST Flags_Bwd RST Flags_bin_0'
#  'Bwd RST Flags_Bwd RST Flags_bin_1'
#  'Fwd Header Length_Fwd Header Length_bin_0'
#  'Fwd Header Length_Fwd Header Length_bin_1'
#  'Fwd Header Length_Fwd Header Length_bin_2'
#  'Bwd Header Length_Bwd Header Length_bin_0'
#  'Bwd Header Length_Bwd Header Length_bin_1'
#  'Bwd Header Length_Bwd Header Length_bin_2'
#  'Fwd Packets/s_Fwd Packets/s_bin_0' 'Fwd Packets/s_Fwd Packets/s_bin_1'
#  'Fwd Packets/s_Fwd Packets/s_bin_2' 'Fwd Packets/s_Fwd Packets/s_bin_3'
#  'Bwd Packets/s_Bwd Packets/s_bin_0' 'Bwd Packets/s_Bwd Packets/s_bin_1'
#  'Bwd Packets/s_Bwd Packets/s_bin_2' 'Bwd Packets/s_Bwd Packets/s_bin_3'
#  'Packet Length Min_Packet Length Min_bin_0'
#  'Packet Length Min_Packet Length Min_bin_1'
#  'Packet Length Min_Packet Length Min_bin_2'
#  'Packet Length Min_Packet Length Min_bin_3'
#  'Packet Length Max_Packet Length Max_bin_0'
#  'Packet Length Max_Packet Length Max_bin_1'
#  'Packet Length Max_Packet Length Max_bin_2'
#  'Packet Length Max_Packet Length Max_bin_3'
#  'Packet Length Mean_Packet Length Mean_bin_0'
#  'Packet Length Mean_Packet Length Mean_bin_1'
#  'Packet Length Mean_Packet Length Mean_bin_2'
#  'Packet Length Mean_Packet Length Mean_bin_3'
#  'Packet Length Std_Packet Length Std_bin_0'
#  'Packet Length Std_Packet Length Std_bin_1'
#  'Packet Length Std_Packet Length Std_bin_2'
#  'Packet Length Std_Packet Length Std_bin_3'
#  'Packet Length Variance_Packet Length Variance_bin_0'
#  'Packet Length Variance_Packet Length Variance_bin_1'
#  'Packet Length Variance_Packet Length Variance_bin_2'
#  'Packet Length Variance_Packet Length Variance_bin_3'
#  'FIN Flag Count_FIN Flag Count_bin_nan'
#  'SYN Flag Count_SYN Flag Count_bin_0'
#  'SYN Flag Count_SYN Flag Count_bin_1'
#  'RST Flag Count_RST Flag Count_bin_0'
#  'RST Flag Count_RST Flag Count_bin_1'
#  'PSH Flag Count_PSH Flag Count_bin_nan'
#  'ACK Flag Count_ACK Flag Count_bin_0'
#  'ACK Flag Count_ACK Flag Count_bin_1'
#  'CWR Flag Count_CWR Flag Count_bin_nan'
#  'ECE Flag Count_ECE Flag Count_bin_nan'
#  'Down/Up Ratio_Down/Up Ratio_bin_nan'
#  'Average Packet Size_Average Packet Size_bin_0'
#  'Average Packet Size_Average Packet Size_bin_1'
#  'Average Packet Size_Average Packet Size_bin_2'
#  'Average Packet Size_Average Packet Size_bin_3'
#  'Fwd Segment Size Avg_Fwd Segment Size Avg_bin_0'
#  'Fwd Segment Size Avg_Fwd Segment Size Avg_bin_1'
#  'Fwd Segment Size Avg_Fwd Segment Size Avg_bin_2'
#  'Fwd Segment Size Avg_Fwd Segment Size Avg_bin_3'
#  'Bwd Segment Size Avg_Bwd Segment Size Avg_bin_0'
#  'Bwd Segment Size Avg_Bwd Segment Size Avg_bin_1'
#  'Bwd Segment Size Avg_Bwd Segment Size Avg_bin_2'
#  'Bwd Segment Size Avg_Bwd Segment Size Avg_bin_3'
#  'Fwd Bytes/Bulk Avg_Fwd Bytes/Bulk Avg_bin_nan'
#  'Fwd Packet/Bulk Avg_Fwd Packet/Bulk Avg_bin_nan'
#  'Fwd Bulk Rate Avg_Fwd Bulk Rate Avg_bin_nan'
#  'Bwd Bytes/Bulk Avg_Bwd Bytes/Bulk Avg_bin_nan'
#  'Bwd Packet/Bulk Avg_Bwd Packet/Bulk Avg_bin_nan'
#  'Bwd Bulk Rate Avg_Bwd Bulk Rate Avg_bin_nan'
#  'Subflow Fwd Packets_Subflow Fwd Packets_bin_nan'
#  'Subflow Fwd Bytes_Subflow Fwd Bytes_bin_0'
#  'Subflow Fwd Bytes_Subflow Fwd Bytes_bin_1'
#  'Subflow Fwd Bytes_Subflow Fwd Bytes_bin_2'
#  'Subflow Fwd Bytes_Subflow Fwd Bytes_bin_3'
#  'Subflow Bwd Packets_Subflow Bwd Packets_bin_nan'
#  'Subflow Bwd Bytes_Subflow Bwd Bytes_bin_0'
#  'Subflow Bwd Bytes_Subflow Bwd Bytes_bin_1'
#  'Subflow Bwd Bytes_Subflow Bwd Bytes_bin_2'
#  'Subflow Bwd Bytes_Subflow Bwd Bytes_bin_3'
#  'FWD Init Win Bytes_FWD Init Win Bytes_bin_0'
#  'FWD Init Win Bytes_FWD Init Win Bytes_bin_1'
#  'Bwd Init Win Bytes_Bwd Init Win Bytes_bin_nan'
#  'Fwd Act Data Pkts_Fwd Act Data Pkts_bin_0'
#  'Fwd Act Data Pkts_Fwd Act Data Pkts_bin_1'
#  'Fwd Seg Size Min_Fwd Seg Size Min_bin_0'
#  'Fwd Seg Size Min_Fwd Seg Size Min_bin_1'
#  'Active Mean_Active Mean_bin_nan' 'Active Std_Active Std_bin_nan'
#  'Active Max_Active Max_bin_nan' 'Active Min_Active Min_bin_nan'
#  'Idle Mean_Idle Mean_bin_nan' 'Idle Std_Idle Std_bin_nan'
#  'Idle Max_Idle Max_bin_nan' 'Idle Min_Idle Min_bin_nan'
#  'ICMP Code_ICMP Code_bin_nan' 'ICMP Type_ICMP Type_bin_nan'
#  'Total TCP Flow Time_Total TCP Flow Time_bin_0'
#  'Total TCP Flow Time_Total TCP Flow Time_bin_1' 'Label_BENIGN'
#  'Label_Infiltration - Portscan'