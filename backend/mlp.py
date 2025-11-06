"""""
Contém a implementação da MLP.

Aqui você coloca:

Definição da classe MLP

Inicialização dos pesos

Funções de ativação (linear, logística, tangente hiperbólica)

Forward pass e backpropagation

Critérios de parada (erro ou número de iterações)

Objetivo: manter a lógica da rede neural separada do resto da aplicação.

"""
import random
import math

# Inicializa pesos
def inicializar_pesos(n_in, n_hidden, n_out):
    # pesos entrada -> oculta
    W1 = [[random.uniform(-1, 1) for _ in range(n_in)] for _ in range(n_hidden)]
    
    # bias camada oculta
    B1 = [random.uniform(-1, 1) for _ in range(n_hidden)]

    # pesos oculta -> saída
    W2 = [[random.uniform(-1, 1) for _ in range(n_hidden)] for _ in range(n_out)]

    # bias camada de saída
    B2 = [random.uniform(-1, 1) for _ in range(n_out)]

    return W1, B1, W2, B2


# Função logística (sigmoid)
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Derivada da sigmoid (entrada deve ser a saída da sigmoid)
def sigmoid_deriv(x):
    return x * (1 - x)


def forward_pass(X, W1, B1, W2, B2):
    # camada oculta
    hidden_raw = []
    hidden_activated = []

    for j in range(len(W1)):  
        soma = 0
        for i in range(len(X)):
            soma += X[i] * W1[j][i]
        soma += B1[j]

        hidden_raw.append(soma)
        hidden_activated.append(sigmoid(soma))

    # camada de saída
    output_raw = []
    output_activated = []

    for k in range(len(W2)):
        soma = 0
        for j in range(len(hidden_activated)):
            soma += hidden_activated[j] * W2[k][j]
        soma += B2[k]

        output_raw.append(soma)
        output_activated.append(sigmoid(soma))

    return hidden_activated, output_activated

def treinar_epoca(X_train, Y_train, W1, B1, W2, B2, taxa=0.1):
    """
    Executa UMA época de treino (varre todas as amostras),
    chama forward_pass + backpropagation para cada amostra,
    atualiza pesos in-place e retorna o erro total da época.
    """
    erro_total = 0.0
    for X, y in zip(X_train, Y_train):
        # forward
        hidden, output = forward_pass(X, W1, B1, W2, B2)

        # backpropagation atualiza W1,B1,W2,B2 in-place e retorna mse da amostra
        mse = backpropagation(X, y, hidden, output, W1, B1, W2, B2, taxa)
        erro_total += mse

    return erro_total  # soma dos MSEs (ou média se preferir)

def backpropagation(X, y, hidden, output, W1, B1, W2, B2, taxa):
    # 1) erro na saída
    erro_saida = [y[k] - output[k] for k in range(len(y))]

    # 2) gradiente da saída (delta)
    delta_saida = [erro_saida[k] * sigmoid_deriv(output[k]) for k in range(len(output))]

    # 3) erro na camada oculta
    erro_oculta = []
    for j in range(len(hidden)):
        soma = sum(delta_saida[k] * W2[k][j] for k in range(len(delta_saida)))
        erro_oculta.append(soma)

    # 4) gradiente da oculta (delta)
    delta_oculta = [erro_oculta[j] * sigmoid_deriv(hidden[j]) for j in range(len(hidden))]

    # 5) atualizar pesos W2 e B2
    for k in range(len(W2)):
        for j in range(len(W2[k])):
            W2[k][j] += taxa * delta_saida[k] * hidden[j]
        B2[k] += taxa * delta_saida[k]

    # 6) atualizar pesos W1 e B1
    for j in range(len(W1)):
        for i in range(len(W1[j])):
            W1[j][i] += taxa * delta_oculta[j] * X[i]
        B1[j] += taxa * delta_oculta[j]

    # retorna erro quadrático médio
    mse = sum(e**2 for e in erro_saida) / len(erro_saida)
    return mse

def treinar(X_train, Y_train, n_in, n_hidden, n_out, taxa=0.1, epocas=5000):
    W1, B1, W2, B2 = inicializar_pesos(n_in, n_hidden, n_out)

    for epoca in range(epocas):
        erro_total = 0

        for X, y in zip(X_train, Y_train):
            # FORWARD
            hidden, output = forward_pass(X, W1, B1, W2, B2)

            # Calcular erro saída
            erro_saida = [y[i] - output[i] for i in range(len(y))]
            erro_total += sum(e**2 for e in erro_saida) / 2

            # Gradiente Saída
            delta_saida = [erro_saida[i] * sigmoid_deriv(output[i]) for i in range(len(y))]

            # Gradiente Oculta
            delta_oculta = []
            for j in range(n_hidden):
                erro_j = sum(delta_saida[k] * W2[k][j] for k in range(n_out))
                delta_oculta.append(erro_j * sigmoid_deriv(hidden[j]))

            # Atualizar pesos W2
            for k in range(n_out):
                for j in range(n_hidden):
                    W2[k][j] += taxa * delta_saida[k] * hidden[j]
                B2[k] += taxa * delta_saida[k]

            # Atualizar pesos W1
            for j in range(n_hidden):
                for i in range(n_in):
                    W1[j][i] += taxa * delta_oculta[j] * X[i]
                B1[j] += taxa * delta_oculta[j]

        if epoca % 500 == 0:
            print(f"Época {epoca}, Erro = {erro_total:.4f}")

    return W1, B1, W2, B2