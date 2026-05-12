import numpy as np


class ScratchMLP:
    def __init__(self, layers, lr=0.01, clip_value=5.0):
        self.layers = layers
        self.lr = lr
        self.clip_value = clip_value   # Gradient clipping eşiği

        self.weights = []
        self.biases = []
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            self.weights.append(w)
            self.biases.append(np.zeros((1, layers[i+1])))

    def _relu(self, x):
        return np.maximum(0.0, x)

    def _relu_deriv(self, x):
        return (x > 0).astype(float)

    def forward(self, X):
        self.activations = [X]
        self.nets = []
        n_hidden = len(self.weights) - 1

        for i in range(n_hidden):
            net = self.activations[-1] @ self.weights[i] + self.biases[i]
            self.nets.append(net)
            self.activations.append(self._relu(net))

        # Son katman: Linear
        net_out = self.activations[-1] @ self.weights[-1] + self.biases[-1]
        self.nets.append(net_out)
        self.activations.append(net_out)
        return self.activations[-1]

    def backward(self, X, y, output):
        n = len(self.weights)
        deltas = [None] * n

        # Çıkış katmanı (linear → türev=1)
        deltas[-1] = -(y - output)

        # Gizli katmanlar
        for i in range(n - 2, -1, -1):
            grad = deltas[i+1] @ self.weights[i+1].T
            grad = np.clip(grad, -self.clip_value, self.clip_value)
            deltas[i] = grad * self._relu_deriv(self.nets[i])

        # Ağırlık güncellemesi
        for i in range(n):
            dw = self.activations[i].T @ deltas[i]
            db = np.sum(deltas[i], axis=0, keepdims=True)
            # Ağırlık güncellemesini de kliple
            dw = np.clip(dw, -self.clip_value, self.clip_value)
            db = np.clip(db, -self.clip_value, self.clip_value)
            self.weights[i] -= self.lr * dw
            self.biases[i]  -= self.lr * db