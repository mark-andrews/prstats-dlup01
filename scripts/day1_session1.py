import numpy
import numpy as np
import matplotlib.pyplot as plt

# 3 input neurons
x = numpy.array([1.5, 2.0, -1.0])
# weights from input layer
w = numpy.random.randn(3) 
b = numpy.random.randn(1)

z = numpy.dot(w, x) + b # linear, aka weighted sum, of x

def sigmoid(z):
    return 1/(1 + np.exp(-z))

# Artifical neuron
sigmoid(numpy.dot(w, x) + b)
np.dot(w, x)
w @ x
sigmoid(w @ x + b)

def tanh(z):
    return np.tanh(z)

def relu(z):
    return np.maximum(0, z)

Z = np.linspace(-5, 5, 100)

plt.plot(Z, sigmoid(Z), label = 'sigmoid')
plt.plot(Z, tanh(Z), label = 'tanh')
plt.plot(Z, relu(Z), label = 'tanh')
plt.legend()


I = 10
H = 5
J = 3

W1 = np.random.randn(I, H)
b1 = np.random.rand(H)

W2 = np.random.randn(H, J)
b2 = np.random.randn(J)

# random input
n = 25 # e.g. n images
X = np.random.rand(n, I)

h = relu(X @ W1 + b1) # hidden layer representation of X
output = sigmoid(h @ W2 + b2)
