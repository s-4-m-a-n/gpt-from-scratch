import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        
        n_hidden_layers = len(weights)-1
        # ======= hidden layer forwardpass =====
        h_out = x
        for l in range(n_hidden_layers):
            h_out = h_out @ weights[l] + biases[l]
            h_out = np.maximum(0, h_out)
        # ======= output layer ==================
        out = h_out @ weights[-1] + biases[-1] 
        # =======================================
        return np.round(out, 5)
