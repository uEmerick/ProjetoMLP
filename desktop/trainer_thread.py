from PySide6.QtCore import QThread, Signal
from mlp import inicializar_pesos, treinar_epoca, forward_pass
import numpy as np
import threading
import time


class TrainerThread(QThread):
    progresso = Signal(int, float)          # (época, erro_médio)
    finalizou = Signal(tuple)               # (W1, B1, W2, B2)
    plato_detected = Signal(int, float)     # (época, erro_médio) -> notificar GUI para decisão

    def __init__(self, X, y, n_hidden=None, epocas=2000, taxa=0.01, erro_alvo=0.0, ativacao_tipo="logistica", plateau_window=10, plateau_std_threshold=1e-5):
        super().__init__()
        self.X = X
        self.y = y
        self.n_hidden = n_hidden  # if None caller should compute and pass
        self.epocas = epocas
        self.taxa = taxa
        self.erro_alvo = erro_alvo
        self.ativacao_tipo = ativacao_tipo

        # plateau detection config
        self.plateau_window = plateau_window
        self.plateau_std_threshold = plateau_std_threshold

        # coordination with main thread when plateau happens
        self._decision_event = threading.Event()
        self._decision_choice = None  # 'continue', 'reduce', 'stop'

        self._stop_requested = False

    # Called by main thread to give decision when plateau requested
    def set_plateau_decision(self, choice):
        """
        choice: 'continue' | 'reduce' | 'stop'
        """
        self._decision_choice = choice
        # wake the waiting thread
        self._decision_event.set()

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        # Calcula dimensões de entrada e saída
        n_in = len(self.X[0])
        # y pode ser one-hot (lista) ou rótulos simples; suporta ambos
        if isinstance(self.y[0], list) or isinstance(self.y[0], (tuple, np.ndarray)):
            n_out = len(self.y[0])
        else:
            n_out = 1

        # se n_hidden não fornecido, calcula heurística
        if self.n_hidden is None:
            self.n_hidden = (n_in + n_out) // 2
            if self.n_hidden < 1:
                self.n_hidden = 1

        # inicializa pesos
        W1, B1, W2, B2 = inicializar_pesos(n_in, self.n_hidden, n_out)

        erros_hist = []  # histórico de erros por época

        for epoca in range(1, self.epocas + 1):
            if self._stop_requested:
                break

            erro_medio = treinar_epoca(self.X, self.y, W1, B1, W2, B2, self.taxa, self.ativacao_tipo)
            erros_hist.append(erro_medio)

            # emitir progresso
            self.progresso.emit(epoca, erro_medio)

            # critério de parada por erro alvo
            if self.erro_alvo is not None and self.erro_alvo > 0 and erro_medio <= self.erro_alvo:
                # atingiu erro desejado -> finaliza
                break

            # verificar platô: se tivermos pelo menos 'plateau_window' épocas,
            # calcular desvio padrão do último bloco e comparar com threshold
            if len(erros_hist) >= self.plateau_window:
                window = erros_hist[-self.plateau_window:]
                std = float(np.std(window))
                # se desvio padrão muito pequeno -> consideramos platô
                if 0.0 <= std <= self.plateau_std_threshold:
                    # emitir sinal pra GUI e aguardar decisão do usuário
                    self.plato_detected.emit(epoca, erro_medio)

                    # aguardar decisão (main thread deverá chamar set_plateau_decision)
                    self._decision_event.clear()
                    # Espera até que a main thread escolha uma ação
                    self._decision_event.wait()

                    # se pediu parar
                    if self._decision_choice == 'stop':
                        self._stop_requested = True
                        break
                    elif self._decision_choice == 'reduce':
                        # reduz taxa em 10%
                        self.taxa *= 0.9
                        # limpa para próxima iteração
                        self._decision_choice = None
                        # continua treinamento (se continuar, pode detectar novamente)
                        continue
                    else:
                        # 'continue' -> apenas continuar sem alterar taxa
                        self._decision_choice = None
                        continue

            # inserir pequena pausa (não estrangula CPU) - opcional
            # time.sleep(0.0001)

        # fim do loop de treinamento: emitir sinais de finalização com pesos
        self.finalizou.emit((W1, B1, W2, B2))