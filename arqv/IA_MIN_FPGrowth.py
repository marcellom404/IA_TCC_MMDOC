
from arqv.IA_mineracao import FPGrowthMiner
from DATA_load import df,treinamento,validacao,alvo,teste
# 

if __name__ == '__main__':
    t = FPGrowthMiner()
    t.generate_rules(df=teste, target_column=alvo, max_rules=1000000, timeout_seconds=60*60*24*360, min_support=0.7)