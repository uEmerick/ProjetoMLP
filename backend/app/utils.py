"""""
Contém funções auxiliares que não fazem parte diretamente da MLP.
Normalização de dados

Divisão treino/teste

Cálculo de matriz de confusão

Detecção de platô e ajuste da taxa de aprendizado

Leitura de CSV e pré-processamento
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def ler_csv(caminho_arquivo: str):
    """Lê um arquivo CSV e retorna um DataFrame pandas."""
    try:
        df = pd.read_csv(caminho_arquivo, sep=',')
        print("\nArquivo CSV lido com sucesso!\n")
        print(df.head())  # mostra as primeiras linhas
        print(f"\nDimensões do DataFrame: {df.shape}\n")
        return df
    except Exception as e:
        print(f"\nErro ao ler o arquivo CSV: {e}\n")
        return None

def preparar_dados(df):
    """Separa as entradas (X) e saídas (y) do DataFrame."""
    try:
        X = df.iloc[:, :-1].values  # todas as colunas menos a última
        y = df.iloc[:, -1].values   # última coluna (classe)
        print(f"\nDados preparados!")
        print(f"Entradas (X): {X.shape}")
        print(f"Saídas (y): {y.shape}")
        print(f"Classes únicas: {set(y)}\n")
        return X, y
    except Exception as e:
        print(f"Erro ao preparar dados: {e}")
        return None, None

def normalizar_dados(X):
    """Normaliza os valores de entrada entre 0 e 1."""
    try:
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_norm = (X - X_min) / (X_max - X_min)
        print("Dados normalizados entre 0 e 1!\n")
        return X_norm
    except Exception as e:
        print(f"Erro ao normalizar dados: {e}")
        return X
    
def dividir_treino_teste(X, y, test_size=0.3, random_state=42):
    """Divide os dados em conjuntos de treino (70%) e teste (30%)"""
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"Dados divididos em treino (70%) e teste (30%)")
        print(f"Tamanho treino: {X_train.shape}, Tamanho teste: {X_test.shape}\n")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Erro ao dividir treino/teste: {e}")
        return None, None, None, None

def codificar_classes(y):
    """Codifica as classes de saída usando One-Hot Encoding."""
    try:
        encoder = OneHotEncoder(sparse_output=False)
        y_encoded = encoder.fit_transform(y.reshape(-1, 1))
        print(f"Classes codificadas com One-Hot Encoding!")
        print(f"Formato original: {y.shape}  →  Codificado: {y_encoded.shape}\n")
        return y_encoded, encoder
    except Exception as e:
        print(f"Erro ao codificar classes: {e}")
        return None, None

# Bloco principal (roda só se o arquivo for executado diretamente)
if __name__ == "__main__":
    caminho = "base_treinamento.csv"
    df = ler_csv(caminho)
    if df is not None:
        X, y = preparar_dados(df)
        X_norm = normalizar_dados(X)
        y_encoded, encoder = codificar_classes(y)
        dividir_treino_teste(X_norm, y_encoded)