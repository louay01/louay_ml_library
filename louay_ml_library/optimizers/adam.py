import numpy as np
from louay_ml_library.base import Optimizer

class Adam(Optimizer):
    """
    Adam optimizer with adaptive learning rates and momentum.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

    def step(self, layers):
        """
        Performs an Adam update.

        Parameters:
        - layers (np.ndarray): All the layers of the netwrok.
        """
        
        self.t += 1
        for layer in layers:
            layer.v_w = self.beta1 * layer.v_w + (1 - self.beta1) * layer.grad_weights
            layer.v_b = self.beta1 * layer.v_b + (1 - self.beta1) * layer.grad_bias

            layer.s_w = self.beta2 * layer.s_w + (1 - self.beta2) * layer.grad_weights ** 2
            layer.s_b = self.beta2 * layer.s_b + (1 - self.beta2) * layer.grad_bias ** 2

            v_hat_w = layer.v_w / (1 - self.beta1 ** self.t)
            v_hat_b = layer.v_b / (1 - self.beta1 ** self.t)

            s_hat_w = layer.s_w / (1 - self.beta2 ** self.t)
            s_hat_b = layer.s_b / (1 - self.beta2 ** self.t)

            layer.weights -=  self.learning_rate * v_hat_w / (np.sqrt(s_hat_w) + self.epsilon)
            layer.bias -= self.learning_rate * v_hat_b / (np.sqrt(s_hat_b) + self.epsilon)
