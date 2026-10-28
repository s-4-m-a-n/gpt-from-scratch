import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:

        x = np.array([x]) # (n, m)
        W1 = np.array(W1) # (o X m)
        b1 = np.array(b1) # (1 X o)
        W2 = np.array(W2) # (p X o)
        b2 = np.array(b2) # (1 X p)
        y_true = np.array([y_true]) # (n, 1)

        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        z1 = x @ W1.T + b1 # (n, o)
        a1 = np.maximum(0, z1) # (n, o)
        z2 = a1 @ W2.T + b2 # (n, p)

        # Loss: MSE = mean((predictions - y_true)^2)
        loss = np.mean(np.square(z2 - y_true)) # scalar (globle mean) 
        
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        
        n = y_true.shape[0]

        dz2 = 2/n * (z2 - y_true) # (n, p)
        dW2 = dz2.T @ a1 #(p, n) @ (n, o) = (p X o)
        db2 = dz2 # (n, p)

        da1 = dz2 @ W2 # (n, p) @ (p, o) = (n, o)
        da1 = da1.flatten() # (n, o)

        dz1 = da1 * (z1 > 0).astype(float) # (n, o)
        dW1 = dz1.T @ x  # (o, n) @ (n, m) = (o, m)
        db1 = dz1 #(o, n)

        return {
            "loss": np.round(loss, 4),
            "dW1": np.round(dW1, 4),
            "db1": np.round(db1.flatten(), 4),
            "dW2": np.round(dW2, 4),
            "db2": np.round(db2.flatten(), 4)
        }