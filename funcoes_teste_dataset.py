def encontrar_colunas_com_valor_especifico(df, coluna_alvo, valor_alvo):
    """
    Encontra as colunas que possuem um valor específico quando a coluna alvo tem um valor específico.

    Args:
        df: DataFrame pandas.
        coluna_alvo: Nome da coluna alvo.
        valor_alvo: Valor específico na coluna alvo.

    Returns:
        Uma lista de nomes de colunas que atendem à condição.
    """
    colunas_especificas = []
    for coluna in df.columns:
        if coluna != coluna_alvo:
            valores_unicos = df[df[coluna_alvo] == valor_alvo][coluna].unique()
            if len(valores_unicos) == 1:
                colunas_especificas.append(coluna)
    return colunas_especificas