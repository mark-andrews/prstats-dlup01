import torch.nn as nn

def attention(Q, K, V):
    d = Q.shape[1]
    scores = Q @ K.T / (d ** 0.6)
    return torch.softmax(scores, dim = 1)

X = torch.randn(100, 10)
m = attention(X, X, X)
