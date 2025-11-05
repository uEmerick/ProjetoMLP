from utils import ler_csv, preparar_dados, normalizar_dados, codificar_classes, detectar_dimensoes
from mlp import treinar, forward_pass

if __name__ == "__main__":
    caminho = "base_treinamento.csv"
    dados = ler_csv(caminho)

    X, y = preparar_dados(dados)
    X_norm = normalizar_dados(X)
    y_encoded, mapa = codificar_classes(y)
    input_dim, output_dim = detectar_dimensoes(X_norm, y_encoded)

    print(f"Dimensão de entrada: {input_dim}")
    print(f"Número de classes (saídas): {output_dim}")
    print(f"Mapa de classes: {mapa}")

    hidden_dim = (input_dim + output_dim) // 2 or 1
    print(f"Neurônios camada oculta: {hidden_dim}")

    print("\nTreinando a MLP... Aguarde...\n")

    W1, B1, W2, B2 = treinar(
        X_norm, 
        y_encoded,
        n_in=input_dim,
        n_hidden=hidden_dim,
        n_out=output_dim,
        taxa=0.5,
        epocas=5000
    )

    print("\nTestando a MLP com os dados de treinamento:\n")
    for entrada in X_norm:
        _, saida = forward_pass(entrada, W1, B1, W2, B2)
        print(entrada, "->", saida)
