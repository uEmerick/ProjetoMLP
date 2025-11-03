import sys
import pandas as pd
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QFileDialog, QTextEdit, QLabel
)


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
        self.btn_normalizar.clicked.connect(self.normalizar_dados)
        self.btn_normalizar.setEnabled(False)  # só habilita depois de carregar CSV
        self.layout.addWidget(self.btn_normalizar)

        # Área de texto para mostrar dados
        self.texto = QTextEdit()
        self.texto.setReadOnly(True)
        self.layout.addWidget(self.texto)

        # Status
        self.status = QLabel("Nenhum arquivo carregado.")
        self.layout.addWidget(self.status)

        # Variáveis
        self.df = None
        self.X = None
        self.y = None
        self.X_norm = None

    def carregar_csv(self):
        caminho, _ = QFileDialog.getOpenFileName(self, "Selecionar arquivo CSV", "", "CSV Files (*.csv)")
        if caminho:
            try:
                self.df = pd.read_csv(caminho, sep=',')
                self.texto.setPlainText(str(self.df.head()))
                self.status.setText(f"Arquivo carregado: {caminho}\nDimensões: {self.df.shape}")
                self.btn_normalizar.setEnabled(True)
                # Separar entradas e saídas
                self.X = self.df.iloc[:, :-1].values
                self.y = self.df.iloc[:, -1].values
            except Exception as e:
                self.status.setText(f"Erro ao ler CSV: {e}")

    def normalizar_dados(self):
        if self.X is not None:
            try:
                X_min = np.min(self.X, axis=0)
                X_max = np.max(self.X, axis=0)
                self.X_norm = (self.X - X_min) / (X_max - X_min)
                self.status.setText("✅ Dados normalizados entre 0 e 1!")
            except Exception as e:
                self.status.setText(f"Erro ao normalizar: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
