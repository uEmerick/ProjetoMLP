"""
Contém a implementação da MLP (atualizado).

- Funções de ativação: linear, logistica (sigmoid), hiperbolica (tanh)
- forward_pass e backpropagation parametrizados por ativação
- treinar_epoca retorna erro médio da época
- inicializar_pesos mantida
"""

import random
import math

# Inicializa pesos
def inicializar_pesos(n_in, n_hidden, n_out):
    # pesos entrada -> oculta (n_hidden x n_in)
    W1 = [[random.uniform(-1, 1) for _ in range(n_in)] for _ in range(n_hidden)]
    B1 = [random.uniform(-1, 1) for _ in range(n_hidden)]

    # pesos oculta -> saída (n_out x n_hidden)
    W2 = [[random.uniform(-1, 1) for _ in range(n_hidden)] for _ in range(n_out)]
    B2 = [random.uniform(-1, 1) for _ in range(n_out)]

    return W1, B1, W2, B2


# -------------------------
# Funções de ativação
# -------------------------
def ativacao_val(x, tipo="logistica"):
    """Recebe valor escalar x e tipo e retorna ativação."""
    if tipo == "linear":
        return x
    elif tipo == "hiperbolica":
        # tanh
        return math.tanh(x)
    else:
        # logística / sigmoid
        return 1.0 / (1.0 + math.exp(-x))


def derivada_ativacao_por_saida(saida_ativada, tipo="logistica"):
    """
    Recebe a saída já ativada (por exemplo sigmoid(x) ou tanh(x))
    e retorna a derivada em relação a x.
    """
    if tipo == "linear":
        return 1.0
    elif tipo == "hiperbolica":
        # d(tanh)/dx = 1 - tanh^2(x) -> usando saida_ativada
        return 1.0 - (saida_ativada ** 2)
    else:
        # logística: f'(x) = f(x) * (1 - f(x))
        return saida_ativada * (1.0 - saida_ativada)


# -------------------------
# Forward pass parametrizado
# -------------------------
def forward_pass(X, W1, B1, W2, B2, ativacao_tipo="logistica"):
    """
    X: vetor de entrada (lista)
    W1: list of lists (n_hidden x n_in)
    B1: list biases hidden (n_hidden)
    W2: list of lists (n_out x n_hidden)
    B2: list biases out (n_out)
    ativacao_tipo: 'linear' | 'logistica' | 'hiperbolica'
    retorna: (hidden_ativada_list, output_ativada_list)
    """
    # camada oculta
    hidden_raw = []
    hidden_activated = []
    for j in range(len(W1)):
        s = 0.0
        for i in range(len(X)):
            s += X[i] * W1[j][i]
        s += B1[j]
        hidden_raw.append(s)
        hidden_activated.append(ativacao_val(s, ativacao_tipo))

    # camada de saída
    output_raw = []
    output_activated = []
    for k in range(len(W2)):
        s = 0.0
        for j in range(len(hidden_activated)):
            s += hidden_activated[j] * W2[k][j]
        s += B2[k]
        output_raw.append(s)
        output_activated.append(ativacao_val(s, ativacao_tipo))

    return hidden_activated, output_activated


# -------------------------
# Backpropagation parametrizado
# -------------------------
def backpropagation(X, y, hidden, output, W1, B1, W2, B2, taxa, ativacao_tipo="logistica"):
    """
    X: vetor de entrada
    y: vetor alvo (one-hot)
    hidden: lista ativada da camada oculta
    output: lista ativada da camada de saída
    atualiza pesos in-place
    retorna: mse (erro quadrático médio da amostra)
    """
    # erro na saída (y - output)
    erro_saida = [y[k] - output[k] for k in range(len(y))]

    # delta na saída (usando derivada baseada na saída ativada)
    delta_saida = [
        erro_saida[k] * derivada_ativacao_por_saida(output[k], ativacao_tipo)
        for k in range(len(output))
    ]

    # erro na camada oculta (retropropagar)
    erro_oculta = []
    for j in range(len(hidden)):
        soma = 0.0
        for k in range(len(delta_saida)):
            soma += delta_saida[k] * W2[k][j]
        erro_oculta.append(soma)

    # delta na oculta
    delta_oculta = [
        erro_oculta[j] * derivada_ativacao_por_saida(hidden[j], ativacao_tipo)
        for j in range(len(hidden))
    ]

    # atualizar W2 e B2
    for k in range(len(W2)):
        for j in range(len(W2[k])):
            W2[k][j] += taxa * delta_saida[k] * hidden[j]
        B2[k] += taxa * delta_saida[k]

    # atualizar W1 e B1
    for j in range(len(W1)):
        for i in range(len(W1[j])):
            W1[j][i] += taxa * delta_oculta[j] * X[i]
        B1[j] += taxa * delta_oculta[j]

    mse = sum(e ** 2 for e in erro_saida) / len(erro_saida)
    return mse


# -------------------------
# treinar_epoca: varre todas as amostras e chama backpropagation
# -------------------------
def treinar_epoca(X_train, Y_train, W1, B1, W2, B2, taxa=0.1, ativacao_tipo="logistica"):
    """
    Executa UMA época de treino (varre todas as amostras),
    atualiza pesos in-place e retorna o erro médio da época.
    """
    erros = []
    for X, y in zip(X_train, Y_train):
        hidden, output = forward_pass(X, W1, B1, W2, B2, ativacao_tipo)
        mse = backpropagation(X, y, hidden, output, W1, B1, W2, B2, taxa, ativacao_tipo)
        erros.append(mse)

    # retornar erro médio da época
    if len(erros) == 0:
        return 0.0
    return sum(erros) / len(erros)


# -------------------------
# Função utilitária de treino (legacy, não usada pela thread)
# -------------------------
def treinar(X_train, Y_train, n_in, n_hidden, n_out, taxa=0.1, epocas=5000, ativacao_tipo="logistica"):
    W1, B1, W2, B2 = inicializar_pesos(n_in, n_hidden, n_out)
    for epoca in range(epocas):
        erro_medio = treinar_epoca(X_train, Y_train, W1, B1, W2, B2, taxa, ativacao_tipo)
        if epoca % 500 == 0:
            print(f"Época {epoca}, Erro médio = {erro_medio:.6f}")
    return W1, B1, W2, B2