content = """# Projeto MLP (Perceção Multicamadas)

Este projeto consiste numa implementação completa de uma rede neuronal artificial do tipo **Multi-Layer Perceptron (MLP)** desenvolvida em Python. O sistema foi construído com a separação entre a lógica base da rede (construída de raiz) e uma Interface Gráfica de Utilizador (GUI) interativa.

## 🚀 Funcionalidades

* **Implementação da MLP de Raiz**: Lógica de *Forward Pass* e algoritmo de *Backpropagation* desenvolvidos sem o uso de bibliotecas de Deep Learning externas.
* **Funções de Ativação**: Suporte para funções de ativação Linear, Logística (Sigmoid) e Tangente Hiperbólica (Tanh).
* **Processamento de Dados Automático**:
  * Leitura e formatação de ficheiros `.csv` (inclui a base de dados Iris como exemplo).
  * Normalização de dados de entrada entre 0 e 1.
  * Codificação de classes (*One-Hot Encoding*).
  * Divisão automática entre conjuntos de treino e teste.
* **Interface Gráfica de Utilizador (GUI)**: Aplicação desktop desenvolvida em **PySide6**.
* **Visualização de Resultados**:
  * Gráfico de erro por época atualizado.
  * Geração automática de uma Matriz de Confusão após os testes.
* **Sistema de Deteção de Estagnação (Platô)**: Funcionalidade em *background* (utilizando *Threads*) que deteta quando o erro deixa de diminuir significativamente, permitindo ao utilizador decidir se pretende parar o treino, continuar ou reduzir a taxa de aprendizagem.

## 📁 Estrutura do Projeto

O código está dividido em duas pastas principais:

### `backend/` (Lógica e CLI)
Contém o núcleo matemático da rede neuronal e scripts de tratamento de dados.
* `mlp.py`: Funções centrais da rede, incluindo inicialização de pesos, ativação, forward pass e backpropagation.
* `utils.py`: Funções auxiliares para ler dados de CSV, separar dados de treino e teste, e normalizar valores.
* `main.py`: Ponto de entrada para executar o modelo via linha de comandos (CLI).
* `Base_Treinamento_Iris.csv`: Ficheiro de exemplo contendo o famoso *dataset* Iris.

### `desktop/` (Interface Gráfica)
Contém a aplicação Desktop construída sobre o *backend*.
* `main.py`: Janela principal da aplicação (desenhada com PySide6), unindo a lógica aos botões e gráficos (usando Matplotlib).
* `trainer_thread.py`: Implementação da classe `TrainerThread` (QThread), garantindo que o treino da rede corra em segundo plano sem bloquear a interface.

## 🛠️ Tecnologias Utilizadas

* **Linguagem**: Python 3.x
* **Interface Gráfica**: PySide6 (Qt)
* **Gráficos e Avaliação**: Matplotlib, NumPy e Scikit-learn (`confusion_matrix`, `train_test_split`)
* **Estruturas Base**: Módulos embutidos como `math`, `random` e `csv`.

## ⚙️ Como Executar

### 1. Instalar as Dependências
Certifique-se de que tem o Python instalado no seu sistema. De seguida, instale as bibliotecas necessárias:
