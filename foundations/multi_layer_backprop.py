import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # m -> input feature dim
        # p -> output dim = 1 (unbatch) 

        x = np.array(x) # (m, ) # for this setup n = 1
        W1 = np.array(W1) # (o X m)
        b1 = np.array(b1) # (o, )
        W2 = np.array(W2) # (p X o)
        b2 = np.array(b2) # (p, )
        y_true = np.array(y_true) # (p,)

        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        z1 = x @ W1.T + b1 # (o,)
        a1 = np.maximum(0, z1) # (o,)
        z2 = a1 @ W2.T + b2 # (p, )

        # Loss: MSE = mean((predictions - y_true)^2)
        loss = np.mean(np.square(z2 - y_true)) # scalar
        
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        
        n = len(y_true) if y_true.ndim > 0 else 1 #(p, ) or 1
        dz2 = 2/n * (z2 - y_true) # (p,)
        dW2 = dz2.reshape(-1, 1) @ a1.reshape(1, -1) #(p, 1) @ (1, o) = (p X o)
        db2 = dz2 # (p, )

        da1 = dz2.reshape(1, -1) @ W2 # (1, p) @ (p, o) = (1, o)
        da1 = da1.flatten() # (o, )

        dz1 = da1 * (z1 > 0).astype(float) # (o, )
        dW1 = dz1.reshape(-1, 1) @ x.reshape(1, -1)  # (o, 1) @ (1, m) = (o, m)
        db1 = dz1

        return {
            "loss": np.round(loss, 4),
            "dW1": np.round(dW1, 4),
            "db1": np.round(db1, 4),
            "dW2": np.round(dW2, 4),
            "db2": np.round(db2, 4)
        }