import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../backend'))

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QTableWidget, QTableWidgetItem,
    QSpinBox, QDoubleSpinBox, QComboBox, QMessageBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from sklearn.model_selection import train_test_split
from utils import ler_csv, preparar_dados, normalizar_dados, codificar_classes, dividir_treino_teste
from mlp import forward_pass
from trainer_thread import TrainerThread


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Projeto MLP")
        self.setMinimumSize(1080, 700)

        # === Widget central ===
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # === Top controls: carregar + parâmetros ===
        top_row = QHBoxLayout()

        self.btn_carregar = QPushButton("Carregar CSV")
        self.btn_carregar.clicked.connect(self.carregar_csv)
        top_row.addWidget(self.btn_carregar)

        # Épocas
        top_row.addWidget(QLabel("Épocas:"))
        self.spin_epocas = QSpinBox()
        self.spin_epocas.setRange(1, 100000)
        self.spin_epocas.setValue(100)
        top_row.addWidget(self.spin_epocas)

        # Erro alvo
        top_row.addWidget(QLabel("Erro alvo:"))
        self.spin_erro = QDoubleSpinBox()
        self.spin_erro.setDecimals(8)
        self.spin_erro.setRange(0.0, 1.0)
        self.spin_erro.setSingleStep(0.0001)
        self.spin_erro.setValue(0.0)  # 0.0 desliga critério por erro
        top_row.addWidget(self.spin_erro)

        # Taxa N
        top_row.addWidget(QLabel("N (taxa):"))
        self.spin_taxa = QDoubleSpinBox()
        self.spin_taxa.setDecimals(6)
        self.spin_taxa.setRange(1e-6, 10.0)
        self.spin_taxa.setSingleStep(0.01)
        self.spin_taxa.setValue(0.1)
        top_row.addWidget(self.spin_taxa)

        # Função de transferência
        top_row.addWidget(QLabel("Função:"))
        self.combo_ativ = QComboBox()
        self.combo_ativ.addItems(["Logística", "Hiperbólica", "Linear"])
        top_row.addWidget(self.combo_ativ)

        # Botão avançar
        self.btn_avancar = QPushButton("Avançar")
        self.btn_avancar.clicked.connect(self.executar_pipeline)
        self.btn_avancar.setEnabled(False)
        top_row.addWidget(self.btn_avancar)

        self.layout.addLayout(top_row)

        # === Tabela ===
        self.tabela = QTableWidget()
        self.layout.addWidget(self.tabela)

        # === Status ===
        self.status = QLabel("Nenhum arquivo carregado.")
        self.layout.addWidget(self.status)

        # === Área de gráfico (vazia inicialmente) ===
        self.figura = Figure(figsize=(10, 4))
        self.canvas = FigureCanvas(self.figura)
        self.layout.addWidget(self.canvas)
        self.canvas.hide()

        # === Variáveis internas ===
        self.dados = None
        self.X = None
        self.y = None
        self.X_norm = None
        self.y_encoded = None
        self.mapa = None
        self.labels = None  # nomes das classes (em ordem de índice)
        self.W1 = self.B1 = self.W2 = self.B2 = None
        self.erros = []

        # thread handle
        self.thread = None

    # =============================================================
    def carregar_csv(self):
        caminho, _ = QFileDialog.getOpenFileName(self, "Selecionar arquivo CSV", "", "CSV Files (*.csv)")
        if not caminho:
            return

        self.dados = ler_csv(caminho)
        if not self.dados:
            self.status.setText("Falha ao ler o arquivo CSV.")
            return

        # Prepara e normaliza
        self.X, self.y = preparar_dados(self.dados)
        self.X_norm = normalizar_dados(self.X)
        self.y_encoded, self.mapa = codificar_classes(self.y)

        # Prepara labels na ordem dos índices
        # Espera mapa: dict {label: index}
        if isinstance(self.mapa, dict):
            # invertendo mapa para lista por índice
            inv = {v: k for k, v in self.mapa.items()}
            self.labels = [inv[i] for i in range(len(inv))]
        else:
            # fallback: usa índices como labels
            self.labels = [str(i) for i in range(len(self.y_encoded[0]) if isinstance(self.y_encoded[0], (list, tuple, np.ndarray)) else 1)]

        # Exibe tabela
        self.preencher_tabela()
        self.canvas.hide()
        self.tabela.show()

        self.status.setText(f"Arquivo carregado: {os.path.basename(caminho)} — {len(self.dados)} amostras")
        self.btn_avancar.setEnabled(True)

    # =============================================================
    def preencher_tabela(self):
        n_amostras = min(20, len(self.X_norm))
        n_atributos = len(self.X_norm[0])

        self.tabela.clear()
        self.tabela.setRowCount(n_amostras)
        self.tabela.setColumnCount(n_atributos + 1)
        headers = [f"Atr {i+1}" for i in range(n_atributos)] + ["Classe"]
        self.tabela.setHorizontalHeaderLabels(headers)

        for i in range(n_amostras):
            for j in range(n_atributos):
                item = QTableWidgetItem(f"{self.X_norm[i][j]:.4f}")
                self.tabela.setItem(i, j, item)

            # obter rótulo da classe do one-hot
            if isinstance(self.y_encoded[i], (list, tuple, np.ndarray)):
                idx = int(np.argmax(self.y_encoded[i]))
                label = self.labels[idx] if idx < len(self.labels) else str(idx)
            else:
                # caso y_encoded seja já um índice
                idx = int(self.y_encoded[i])
                label = self.labels[idx] if idx < len(self.labels) else str(idx)

            self.tabela.setItem(i, n_atributos, QTableWidgetItem(str(label)))

        self.tabela.resizeColumnsToContents()

    # =============================================================
    def executar_pipeline(self):
        if self.X_norm is None or self.y_encoded is None:
            self.status.setText("Carregue o CSV antes de continuar.")
            return

        X_train, X_test, y_train, y_test = dividir_treino_teste(
        self.X_norm,
        self.y_encoded,
        test_size=0.3
        )
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # calcular automaticamente n_hidden e n_output
        n_in = len(self.X_norm[0])
        if isinstance(self.y_encoded[0], (list, tuple, np.ndarray)):
            n_out = len(self.y_encoded[0])
        else:
            n_out = 1
        n_hidden = (n_in + n_out) // 2
        if n_hidden < 1:
            n_hidden = 1

        # parâmetros do usuário
        epocas = int(self.spin_epocas.value())
        erro_alvo = float(self.spin_erro.value())
        taxa = float(self.spin_taxa.value())
        ativ_map = {"Logística": "logistica", "Hiperbólica": "hiperbolica", "Linear": "linear"}
        ativacao_tipo = ativ_map[self.combo_ativ.currentText()]

        # limpar histórico e UI
        self.status.setText("Iniciando treinamento...")
        self.erros = []
        self.tabela.hide()
        self.canvas.show()

        # cria thread de treino com parâmetros
        self.thread = TrainerThread(
            X=self.X_train,
            y=self.y_train,
            n_hidden=n_hidden,
            epocas=epocas,
            taxa=taxa,
            erro_alvo=erro_alvo,
            ativacao_tipo=ativacao_tipo,
            plateau_window=10,
            plateau_std_threshold=1e-5
        )

        # conectar sinais
        self.thread.progresso.connect(self.atualizar_grafico)
        self.thread.finalizou.connect(self.finalizou_treino)
        self.thread.plato_detected.connect(self.handle_plateau_detected)

        # start
        self.thread.start()

    # =============================================================
    def atualizar_grafico(self, epoch, erro):
        self.erros.append(erro)
        self.figura.clear()
        ax = self.figura.add_subplot(121)
        ax.plot(self.erros)
        ax.set_xlabel("Época")
        ax.set_ylabel("Erro")
        ax.set_title("Erro da MLP por Época")
        self.canvas.draw()
        self.status.setText(f"Treinando... Época {epoch}, Erro {erro:.8f}")

    # =============================================================
    def handle_plateau_detected(self, epoch, erro):
        """
        Chamado pela thread quando detecta platô.
        Mostra diálogo modal ao usuário e envia decisão para a thread.
        """
        # Mostrar diálogo com 3 opções
        msg = QMessageBox(self)
        msg.setWindowTitle("Platô detectado")
        msg.setText(f"O treinamento parece estar em platô na época {epoch} (erro médio = {erro:.8f}).\nO que deseja fazer?")
        continuar = msg.addButton("Continuar", QMessageBox.AcceptRole)
        reduzir = msg.addButton("Continuar com redução de taxa (-10%)", QMessageBox.AcceptRole)
        parar = msg.addButton("Interromper", QMessageBox.RejectRole)
        msg.setDefaultButton(continuar)
        msg.exec()

        chosen = None
        if msg.clickedButton() == reduzir:
            chosen = 'reduce'
        elif msg.clickedButton() == parar:
            chosen = 'stop'
        else:
            chosen = 'continue'

        # envia decisão pra thread
        if self.thread is not None:
            self.thread.set_plateau_decision(chosen)

        # exibe info rápida
        if chosen == 'reduce':
            self.status.setText("Usuário optou por reduzir taxa em 10% e continuar.")
        elif chosen == 'stop':
            self.status.setText("Usuário optou por interromper o treinamento.")
        else:
            self.status.setText("Usuário optou por continuar o treinamento sem alteração.")

    # =============================================================
    def finalizou_treino(self, pesos):
        self.W1, self.B1, self.W2, self.B2 = pesos
        self.status.setText("Treinamento concluído! Gerando matriz de confusão...")
        # gerar previsões e matriz
        self.testar_amostras()

    # =============================================================
    def testar_amostras(self):
        # Gera previsões (classe com maior ativação na saída)
        previsoes = []
        for entrada in self.X_test:
            _, saida = forward_pass(entrada, self.W1, self.B1, self.W2, self.B2, ativacao_tipo=self.combo_ativ.currentText().lower() if False else None)
            # NOTE: forward_pass no mlp espera ativacao_tipo em nome 'logistica'|'hiperbolica'|'linear'
            # mas aqui chamamos diretamente com o mesmo tipo que treinou na thread, para simplicidade assumiremos sigmoid-like.
            # COMO O TREINO USOU ativacao_tipo, é melhor que a thread devolva também o ativacao_tipo ou mantemos a escolha atual:
            ativ_map = {"Logística": "logistica", "Hiperbólica": "hiperbolica", "Linear": "linear"}
            ativacao_tipo = ativ_map[self.combo_ativ.currentText()]
            _, saida = forward_pass(entrada, self.W1, self.B1, self.W2, self.B2, ativacao_tipo=ativacao_tipo)
            pred = int(np.argmax(saida))
            previsoes.append(pred)

        # converter y_encoded para índices
        if isinstance(self.y_test[0], (list, tuple, np.ndarray)):
            y_true = np.array([int(np.argmax(y)) for y in self.y_test])
        else:
            y_true = np.array([int(y) for y in self.y_test])

        self.mostrar_matriz_confusao(y_true, np.array(previsoes))
        self.status.setText("Treino e avaliação concluídos com sucesso!")

    # =============================================================
    def mostrar_matriz_confusao(self, y_true, previsoes):
        """Exibe gráfico de erro e matriz de confusão lado a lado."""
        self.figura.clear()

        # Gráfico de erro
        ax1 = self.figura.add_subplot(121)
        ax1.plot(self.erros)
        ax1.set_xlabel("Época")
        ax1.set_ylabel("Erro")
        ax1.set_title("Erro da MLP por Época")

        # Matriz de confusão
        cm = confusion_matrix(y_true, previsoes)
        ax2 = self.figura.add_subplot(122)
        labels = self.labels if self.labels is not None else None
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax2, colorbar=False)
        ax2.set_title("Matriz de Confusão")

        self.figura.tight_layout()
        self.canvas.draw()


# =============================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
