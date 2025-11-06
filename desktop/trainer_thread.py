from PySide6.QtCore import QThread, Signal
from mlp import inicializar_pesos, treinar_epoca


class TrainerThread(QThread):
    progresso_signal = Signal(int, float)   # emite (época, erro)
    fim_signal = Signal(object)             # emite (W1, B1, W2, B2)

    def __init__(self, X, y, n_hidden=6, epocas=500, taxa=0.1, parent=None):
        super().__init__(parent)
        self.X = X
        self.y = y
        self.n_hidden = n_hidden
        self.epocas = epocas
        self.taxa = taxa
        self._stop_requested = False

    def run(self):
        n_in = len(self.X[0])
        n_out = len(self.y[0]) if isinstance(self.y[0], list) else 1

        W1, B1, W2, B2 = inicializar_pesos(n_in, self.n_hidden, n_out)

        for ep in range(1, self.epocas + 1):
            if self._stop_requested:
                break

            erro_epoca = treinar_epoca(self.X, self.y, W1, B1, W2, B2, taxa=self.taxa)
            self.progresso_signal.emit(ep, erro_epoca)

        self.fim_signal.emit((W1, B1, W2, B2))

    def stop(self):
        self._stop_requested = True
