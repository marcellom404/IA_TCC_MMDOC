import sqlite3
import json
from datetime import datetime

def get_db_connection():
    """Cria e retorna uma conexão com o banco de dados SQLite."""
    conn = sqlite3.connect('results.db')
    conn.row_factory = sqlite3.Row
    return conn

def create_table():
    """Cria a tabela de resultados se ela não existir."""
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            algorithm TEXT NOT NULL,
            accuracy REAL NOT NULL,
            parameters TEXT,
            timestamp DATETIME NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def save_result(algorithm, accuracy, parameters):
    """Salva um novo resultado no banco de dados."""
    conn = get_db_connection()
    timestamp = datetime.now()
    # Convertendo o dicionário de parâmetros para uma string JSON
    parameters_json = json.dumps(parameters)
    conn.execute(
        'INSERT INTO results (algorithm, accuracy, parameters, timestamp) VALUES (?, ?, ?, ?)',
        (algorithm, accuracy, parameters_json, timestamp)
    )
    conn.commit()
    conn.close()

if __name__ == '__main__':
    # Cria a tabela ao executar o script diretamente
    create_table()
    print("Banco de dados e tabela 'results' criados com sucesso.")
