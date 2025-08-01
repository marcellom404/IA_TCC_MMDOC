from IA_mineracao import AprioriMiner
from IA_mineracao import FPGrowthMiner

from DATA_load import df,treinamento,validacao,teste,alvo
# 

if __name__ == '__main__':
    teste = AprioriMiner()
    teste.generate_rules(df=df, target_column=alvo, max_rules=10, timeout_seconds=60, min_support=0.2)