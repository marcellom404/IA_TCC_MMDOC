import os
import shutil
import sqlite3

def move_dataset(dataset_path, destination_folder="datasets"):
    """
    Move um dataset para uma pasta comum e retorna o novo caminho.

    Args:
        dataset_path: Caminho para o dataset original.
        destination_folder: Nome da pasta de destino (padrão: "datasets").

    Returns:
        O novo caminho do dataset ou None se o arquivo não existir ou ocorrer um erro.
    """
    if not os.path.exists(dataset_path):
        print(f"Erro: O arquivo {dataset_path} não existe.")
        return None

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    dataset_filename = os.path.basename(dataset_path)
    new_dataset_path = os.path.join(destination_folder, dataset_filename)

    try:
        shutil.move(dataset_path, new_dataset_path)
        print(f"Dataset movido para: {new_dataset_path}")
        return new_dataset_path
    except Exception as e:
        print(f"Erro ao mover o dataset: {e}")
        return None


def save_dataset_path_to_db(dataset_path, db_name="dataset_paths.db"):
    """
    Salva o caminho do dataset em um banco de dados SQLite3.

    Args:
        dataset_path: O caminho do dataset.
        db_name: Nome do banco de dados (padrão: "dataset_paths.db").
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Cria a tabela se ela não existir
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE
        )
    ''')

    try:
        cursor.execute("INSERT INTO datasets (path) VALUES (?)", (dataset_path,))
        conn.commit()
        print(f"Caminho do dataset salvo no banco de dados: {dataset_path}")
    except sqlite3.IntegrityError:
        print(f"Caminho do dataset já existe no banco de dados: {dataset_path}")
    finally:
        conn.close()
def get_dataset_paths_from_db(db_name="dataset_paths.db"):
    """
    Recupera todos os caminhos de datasets salvos em um banco de dados SQLite3.

    Args:
        db_name: Nome do banco de dados (padrão: "dataset_paths.db").

    Returns:
        Uma lista de caminhos de datasets ou uma lista vazia se nenhum caminho for encontrado.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Consulta todos os caminhos da tabela datasets
    cursor.execute("SELECT path FROM datasets")
    paths = cursor.fetchall()

    # Fecha a conexão
    conn.close()

    # Retorna uma lista de caminhos
    return [path[0] for path in paths] 