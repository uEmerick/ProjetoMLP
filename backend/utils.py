"""""
Contém funções auxiliares que não fazem parte diretamente da MLP.
Normalização de dados

Divisão treino/teste

Cálculo de matriz de confusão

Detecção de platô e ajuste da taxa de aprendizado

Leitura de CSV e pré-processamento
"""
import csv
from sklearn.model_selection import train_test_split


def ler_csv(caminho):
    dados = []
    try:
        with open(caminho, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # pula cabeçalho
            for linha in reader:
                if len(linha) < 2: 
                    continue

                valores = list(map(float, linha[:-1]))  # entradas
                classe = linha[-1]                     # última coluna = classe
                dados.append((valores, classe))
    except Exception as e:
        print(f"Erro ao ler CSV: {e}")
    
    return dados


def preparar_dados(dados):
    """Separa as entradas e classes a partir da lista lida do CSV."""
    X = [linha[0] for linha in dados]
    y = [linha[1] for linha in dados]

    print(f"Total de amostras: {len(X)}")
    print(f"Exemplo X[0]: {X[0]}")
    print(f"Exemplo Y[0]: {y[0]}")
    print(f"Classes únicas: {set(y)}\n")

    return X, y

def normalizar_dados(X):
    """Normaliza os valores de entrada entre 0 e 1."""
    X_norm = []
    num_features = len(X[0])

    # calcular min/max de cada coluna
    mins = [min(coluna) for coluna in zip(*X)]
    maxs = [max(coluna) for coluna in zip(*X)]

    # normalizar: (x - min) / (max - min)
    for linha in X:
        nova_linha = []
        for i in range(num_features):
            if maxs[i] - mins[i] == 0:  # evitar divisão por zero
                nova_linha.append(0)
            else:
                nova_linha.append((linha[i] - mins[i]) / (maxs[i] - mins[i]))
        X_norm.append(nova_linha)

    return X_norm

def codificar_classes(y):
    classes = sorted(set(y))
    class_to_index = {c: i for i, c in enumerate(classes)}

    y_encoded = []
    for label in y:
        vetor = [0] * len(classes)
        vetor[class_to_index[label]] = 1
        y_encoded.append(vetor)

    return y_encoded, class_to_index

def detectar_dimensoes(X, y_encoded):
    input_dim = len(X[0])
    output_dim = len(y_encoded[0])
    return input_dim, output_dim

def dividir_treino_teste(X, y, test_size=0.3, random_state=42):
    """Divide os dados em conjuntos de treino e teste."""
    try:
        # TENTATIVA com estratificação (ideal para classificação)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
    except Exception as e:
        print(f"Stratify falhou ({e}). Tentando sem estratificação...")
        # FALLBACK sem stratify
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
        )

    print(f"Treino: {len(X_train)} amostras  |  Teste: {len(X_test)} amostras\n")
    return X_train, X_test, y_train, y_test
