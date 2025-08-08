from IA_mineracao import AprioriMiner
from IA_mineracao import create_random_datasets
from DATA_load import treinamento
import gc

alvo_pre = "Label"
alvo_post = "Label_Infiltration"

if __name__ == '__main__':
    miner = AprioriMiner()

    # --- Parâmetros da Busca ---
    support_levels = [0.05, 0.05,0.05, 0.05,0.05, 0.05]
    min_confidence_threshold = 0.7
    num_columns_per_dataset = 40
    num_datasets_to_try = 20 
    max_len = 5

    # Loop principal: itera para criar e testar um dataset por vez.
    for i in range(num_datasets_to_try):
        print("\n" + "="*80)
        print(f"INICIANDO BUSCA NO DATASET ALEATÓRIO Nº {i+1}/{num_datasets_to_try}")
        print("="*80 + "\n")

        # Cria UM dataset aleatório para esta iteração.
        dataset_sample = create_random_datasets(
            original_df=treinamento, 
            target_column=alvo_pre, 
            n_columns_per_dataset=num_columns_per_dataset, 
            n_datasets=1
        )[0]
        
        print(f"Colunas nesta amostra: {list(dataset_sample.columns)}")

        # Loop secundário: itera sobre os diferentes níveis de suporte para o dataset atual.
        for support in support_levels:
            print(f"--- Tentando com Suporte Mínimo: {support:.4f} ---")
            
            miner.generate_rules(
                df=dataset_sample, 
                target_column=alvo_post,
                max_rules=200,
                timeout_seconds=60 * 20,
                min_support=support,
                max_itemset_len=max_len,
                min_confidence=min_confidence_threshold
            )

        # Libera a memória 
        del dataset_sample
        gc.collect()

