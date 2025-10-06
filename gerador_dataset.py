
import pandas as pd
import os

# --- CONFIGURAÇÃO ---
# Coloque aqui os caminhos para os arquivos CSV e as regras de amostragem.

CONFIG = {
    "output_filename": "merged_dataset.csv",
    "min_attack_samples": 100,  # Se um ataque tiver menos amostras que isso, será ignorado.
    "files": {
        "datasets/1/CSECICIDS2018_improved/Friday-02-03-2018.csv": {
            "BENIGN": 10000,
            "attacks": {
                "Botnet Ares": 10000,
            }
        },
        "datasets/1/CSECICIDS2018_improved/Friday-16-02-2018.csv": {
            "BENIGN": 10000,
            "attacks": {
                "DoS Hulk": 10000,
                "FTP-BruteForce - Attempted": 5000
            }
        },
        "datasets/1/CSECICIDS2018_improved/Friday-23-02-2018.csv": {
            "BENIGN": 10000,
            "attacks": {
                # Ataques com poucas amostras...
                "Web Attack - XSS": 73,
                "Web Attack - Brute Force": 62,
            }
        },
        "datasets/1/CSECICIDS2018_improved/Thursday-01-03-2018.csv": {
            "BENIGN": 10000,
            "attacks": {
                "Infiltration - NMAP Portscan": 10000,
            }
        },
        "datasets/1/CSECICIDS2018_improved/Thursday-15-02-2018.csv": {
            "BENIGN": 10000,
            "attacks": {
                "DoS GoldenEye": 10000,
                "DoS Slowloris": 8000,
            }
        },
        "datasets/1/CSECICIDS2018_improved/Thursday-22-02-2018.csv": {
            "BENIGN": 10000,
            "attacks": {
                "Web Attack - Brute Force": 69,
                "Web Attack - XSS": 40,
            }
        },
        "datasets/1/CSECICIDS2018_improved/Tuesday-20-02-2018.csv": {
            "BENIGN": 10000,
            "attacks": {
                "DDoS-LOIC-HTTP": 10000,
            }
        },
        "datasets/1/CSECICIDS2018_improved/Wednesday-14-02-2018.csv": {
            # Este arquivo não tem BENIGN nos dados de exemplo, adicione se necessário
            "BENIGN": 0,
            "attacks": {
                "FTP-BruteForce - Attempted": 10000,
                "SSH-BruteForce": 10000,
            }
        },
        "datasets/1/CSECICIDS2018_improved/Wednesday-21-02-2018.csv": {
            "BENIGN": 10000,
            "attacks": {
                "DDoS-HOIC": 10000,
            }
        },
        "datasets/1/CSECICIDS2018_improved/Wednesday-28-02-2018.csv": {
            "BENIGN": 10000,
            "attacks": {
                "Infiltration - NMAP Portscan": 10000,
            }
        }
    }
}

# --- FIM DA CONFIGURACAO ---

def merge_datasets(config):
    """
    Realiza a fusão de múltiplos datasets CSV com base nas configurações fornecidas.
    """
    all_dfs = []
    label_column = "Label"

    for file_path, rules in config["files"].items():
        if not os.path.exists(file_path):
            print(f"AVISO: Arquivo não encontrado, pulando: {file_path}")
            continue

        print(f"Processando arquivo: {file_path}")
        df = pd.read_csv(file_path)

        # 1. Amostragem de dados BENIGN
        n_benign = rules.get("BENIGN", 0)
        if n_benign > 0:
            benign_df = df[df[label_column] == "BENIGN"]
            if len(benign_df) > 0:
                all_dfs.append(benign_df.sample(n=min(n_benign, len(benign_df)), random_state=64))
                print(f"  - Adicionando {min(n_benign, len(benign_df))}" + " amostras de BENIGN.")

        # 2. Amostragem de ataques
        attack_rules = rules.get("attacks", {})
        for attack_name, n_samples in attack_rules.items():
            attack_df = df[df[label_column] == attack_name]
            
            if len(attack_df) < config["min_attack_samples"]:
                print(f"  - Ignorando ataque '{attack_name}' por ter poucas amostras ({len(attack_df)}).")
                continue

            if len(attack_df) > 0:
                all_dfs.append(attack_df.sample(n=min(n_samples, len(attack_df)), random_state=64))
                print(f"  - Adicionando {min(n_samples, len(attack_df))}" + f" amostras de '{attack_name}'.")

    if not all_dfs:
        print("Nenhum dado foi processado. O arquivo de saída não será gerado.")
        return

    # 3. Combinar e salvar
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    # Embaralhar o dataset final
    merged_df = merged_df.sample(frac=1, random_state=64).reset_index(drop=True)
    
    output_path = config["output_filename"]
    merged_df.to_csv(output_path, index=False)

    print(f"\nDataset mesclado e salvo em: {output_path}")
    print("Distribuição das classes no novo dataset:")
    print(merged_df[label_column].value_counts())

if __name__ == "__main__":
    merge_datasets(CONFIG)
