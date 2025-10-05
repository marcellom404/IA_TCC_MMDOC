# diagnostico.py
from sklearn.metrics import classification_report
# Importar a função de carregamento para o teste de estresse
from DATA_load import get_dados_amostra, alvo as alvo_global

def analisar_resultado(model_name, trained_model, f1_score, attack_metrics, validation_df, alvo_column):
    """
    Analisa o resultado e, se o F1-Score for suspeito, realiza um teste de estresse.
    """
    LIMITE_F1_SUSPEITO = 0.9999

    if f1_score >= LIMITE_F1_SUSPEITO:
        print(f"\n--- DIAGNÓSTICO: F1-SCORE SUSPEITO DETECTADO ---")
        print(f"Modelo: {model_name}")
        print(f"Métricas Originais: {attack_metrics}")
        
        support = attack_metrics.get('support', 'N/A')
        print(f"-> Support Original (amostras de ataque): {support}")

        # --- INÍCIO DO TESTE DE ESTRESSE ---
        print("\n--- INICIANDO TESTE DE ESTRESSE ---")
        print("Gerando um novo conjunto de dados maior para re-avaliação...")
        
        # Pega um novo conjunto de dados grande (o conjunto de treino de uma nova amostra)
        stress_test_df, _, _ = get_dados_amostra()
        
        if stress_test_df.empty or len(stress_test_df) < 2:
            print("Não foi possível gerar dados para o teste de estresse. Abortando.")
            return

        print(f"Re-testando o modelo em {len(stress_test_df)} novas amostras...")

        # Prepara os dados para o teste
        X_stress = stress_test_df.drop(alvo_global, axis=1)
        y_stress_true = stress_test_df[alvo_global]

        # Realiza as previsões no novo conjunto
        y_stress_pred = trained_model.predict(X_stress)

        # Calcula e imprime o novo relatório
        print("\nRelatório de Classificação do Teste de Estresse:")
        stress_report = classification_report(y_stress_true, y_stress_pred, zero_division=0, output_dict=True)
        
        # Imprime o relatório de forma legível
        print(classification_report(y_stress_true, y_stress_pred, zero_division=0))

        # Extrai as métricas de ataque do novo relatório
        stress_attack_metrics = {}
        for label, metrics in stress_report.items():
            if label.upper() != 'BENIGN' and label not in ['accuracy', 'macro avg', 'weighted avg']:
                stress_attack_metrics = metrics
                break
        
        stress_f1 = stress_attack_metrics.get('f1-score', 0)
        
        print(f"\n--- CONCLUSÃO DO TESTE DE ESTRESSE ---")
        print(f"F1-Score Original: {f1_score:.5f}")
        print(f"F1-Score no Teste de Estresse: {stress_f1:.5f}")

        if stress_f1 < 0.95:
            print("-> O desempenho caiu drasticamente. O score original era provavelmente 'sorte' devido a um conjunto de validação pequeno ou muito simples.")
        else:
            print("-> O desempenho permaneceu extremamente alto. Isso reforça a suspeita de VAZAMENTO DE DADOS (DATA LEAKAGE). O modelo aprendeu uma 'regra secreta' que se aplica a todo o dataset.")
            
        print("--- FIM DO DIAGNÓSTICO ---\\n")