import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../backend'))
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QFileDialog, QTextEdit, QLabel
)
from utils import ler_csv, preparar_dados, normalizar_dados, codificar_classes, detectar_dimensoes
from mlp import treinar, forward_pass

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Projeto MLP - Desktop")
        self.setMinimumSize(1080, 600)

        # Widget central
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layout
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Botões
        self.btn_carregar = QPushButton("Carregar CSV")
        self.btn_carregar.clicked.connect(self.carregar_csv)
        self.layout.addWidget(self.btn_carregar)

        self.btn_normalizar = QPushButton("Normalizar Dados")
        self.btn_normalizar.clicked.connect(self.normalizar)
        self.btn_normalizar.setEnabled(False)
        self.layout.addWidget(self.btn_normalizar)

        self.btn_treinar = QPushButton("Treinar MLP")
        self.btn_treinar.clicked.connect(self.treinar_rede)
        self.btn_treinar.setEnabled(False)
        self.layout.addWidget(self.btn_treinar)

        self.btn_testar = QPushButton("Testar Amostras")
        self.btn_testar.clicked.connect(self.testar_amostras)
        self.btn_testar.setEnabled(False)
        self.layout.addWidget(self.btn_testar)

        # Área de texto
        self.texto = QTextEdit()
        self.texto.setReadOnly(True)
        self.layout.addWidget(self.texto)

        # Status
        self.status = QLabel("Nenhum arquivo carregado.")
        self.layout.addWidget(self.status)

        # Variáveis
        self.dados = None
        self.X = None
        self.y = None
        self.X_norm = None
        self.mapa = None
        self.W1 = self.B1 = self.W2 = self.B2 = None

    def carregar_csv(self):
        caminho, _ = QFileDialog.getOpenFileName(self, "Selecionar arquivo CSV", "", "CSV Files (*.csv)")
        if caminho:
            self.dados = ler_csv(caminho)
            if self.dados:
                self.X, self.y = preparar_dados(self.dados)
                self.texto.setPlainText(f"Primeiras 5 amostras:\n{self.dados[:5]}")
                self.status.setText(f"Arquivo carregado: {caminho}\nTotal amostras: {len(self.dados)}")
                self.btn_normalizar.setEnabled(True)

    def normalizar(self):
        if self.X is not None:
            self.X_norm = normalizar_dados(self.X)
            self.status.setText("✅ Dados normalizados entre 0 e 1!")
            self.btn_treinar.setEnabled(True)

    def treinar_rede(self):
        if self.X_norm is None or self.y is None:
            self.status.setText("Carregue e normalize os dados antes de treinar!")
            return

        # Codifica classes
        y_encoded, self.mapa = codificar_classes(self.y)
        input_dim, output_dim = detectar_dimensoes(self.X_norm, y_encoded)
        hidden_dim = (input_dim + output_dim) // 2 or 1

        self.W1, self.B1, self.W2, self.B2 = treinar(
            self.X_norm, y_encoded,
            n_in=input_dim,
            n_hidden=hidden_dim,
            n_out=output_dim,
            taxa=0.5,
            epocas=5000
        )

        self.status.setText(f"Treino concluído! Camada oculta: {hidden_dim} neurônios")
        self.btn_testar.setEnabled(True)

    def testar_amostras(self):
        if not all([self.W1, self.B1, self.W2, self.B2]):
            self.status.setText("Treine a rede antes de testar!")
            return

        resultados = []
        for entrada in self.X_norm:
            _, saida = forward_pass(entrada, self.W1, self.B1, self.W2, self.B2)
            resultados.append(saida)

        # Mostrar os primeiros 5 resultados
        texto_resultados = "\n".join(
            [f"{self.X[i]} -> {resultados[i]}" for i in range(min(5, len(resultados)))]
        )
        self.texto.setPlainText(texto_resultados)
        self.status.setText("✅ Teste realizado!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
