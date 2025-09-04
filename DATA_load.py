
import os
import manipul_arquivo as mov
import funcoes_teste_dataset as teste
import random
import pandas as pd
import kagglehub




global alvo 
alvo = "Label"

if os.path.exists("datasets"):

    path = "datasets/1"
else:
    path = kagglehub.dataset_download("ernie55ernie/improved-cicids2017-and-csecicids2018")
    #move o dataset para fora dos arquivos temporarios, salvando em datasets/
    mov.save_dataset_path_to_db(mov.move_dataset(path))
# informaçoes sobre os datasets no info_dataset.py 'CSECICIDS2018_improved/Friday-02-03-2018.csv'
DATASET_NAME ="thursday.csv"
print(path + "/" + DATASET_NAME)
DATASET_PATH = path + "/" + DATASET_NAME
# Carregando os dados em um DataFrame
DATASET_COLUMN_NAMES = list(pd.read_csv(DATASET_PATH, nrows=1).columns)
DATASET_COLUMN_NAMES.sort(key=str.lower)
# print(f"Colunas do dataset: {DATASET_COLUMN_NAMES}")
num_linhas_total = 6168188  
num_linhas_desejado = 61681  
# Gera uma lista de índices de linhas para pular aleatoriamente pq meu pc nao tem memoria infinita... ainda
skip_indices = sorted(random.sample(range(1, num_linhas_total + 1), 
                                     num_linhas_total - num_linhas_desejado))
# remova o "skiprows=skip_indices" abaixo se for usar o dataset inteiro
df = pd.read_csv(DATASET_PATH,skiprows=skip_indices)


# mostra as colunas
# for i in df.columns.values,df.iloc[1] :
#     print(i)
# for i,j in list(zip(df.columns.values,df.iloc[1] )):
#   print(i,"| De tipo:",str(type(j)).replace("<","").replace(">","").replace("class","").replace("64","").replace("numpy.","").replace("'","").replace(" ",""))
# colunas que precisam ser removidas
columns_to_drop = ["id", "Attempted Category", "Timestamp", "Src IP", "Dst IP","Flow ID","Fwd URG Flags",'Bwd URG Flags', 'URG Flag Count']
df.drop(columns=columns_to_drop, inplace=True)
# print("----------------------------------------",teste.encontrar_colunas_com_valor_especifico(df,alvo,"BENIGN"))

# "label" e a coluna que indica o tipo de fluxo de rede, sendo BENIGN o trafego comum, outros sao algum tipo de tentativa de intrusão
# print(df["Label"].value_counts())

treinamento = df.sample(frac=0.7, random_state=64)
df_restante = df.drop(treinamento.index)
validacao = df_restante.sample(frac=0.6667, random_state=64)
teste = df_restante.drop(validacao.index).sample(frac=0.01,random_state=64)

# print("Total de 4601 linhas no conjunto original, incluindo uma linha com nomes das colunas.")
print(f"Total de {len(df)} linhas no conjunto original")
print(f"Validação: {len(validacao)} linhas")
print(f"Treinamento: {len(treinamento)} linhas")
print(f"Teste: {len(teste)} linhas")
# print("Soma total:", len(validacao) + len(treinamento) + len(teste))

# variaveis apos pre-processamento
# df = pd.read_csv("one_hot_encoded_data.csv")
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
#  'Label_Infiltration - Portscan']