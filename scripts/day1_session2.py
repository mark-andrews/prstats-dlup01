import torch

# torch equivalent of np.array([...])
x = torch.tensor([1.5, 2.0, -1.0])
x.shape
x.dtype

x * x # elementwise multiplication
x @ x # matrix multiplication
x.sum()
 
x
x.abs() # returns a new array
x.abs_() # in place version

# Autograd
x = torch.tensor(2.0, requires_grad = True)
y = x ** 2 # x^2
z = 3 * y # z = 3 x ^ 2
z.backward()
x.grad # dz/dx 


x = torch.tensor(2.0, requires_grad = True)
y = x ** 3 # x^3

import torch.nn.functional as F

h = F.relu(y)
h.backward()
x.grad # dh/dx

x = torch.tensor(2.0, requires_grad = True)
y = x ** 2
z = 10 * y
z.backward()
x.grad # dy/dx

# implement a full feedforward nn
I = 4
H = 3
J = 2

W1 = torch.randn(I, H, requires_grad=True) # input to hidden layer weight matrix
b1 = torch.randn(H, requires_grad=True) # bias/intercept on hidden
W2 = torch.randn(H, J, requires_grad=True) # hidden to output layer weight matrix
b2 = torch.randn(J, requires_grad=True) # bias/intercept on hidden

x_input = torch.randn(I)

hidden = F.relu(x_input @ W1 + b1)
output = hidden @ W2 + b2
loss = (torch.tensor([1, -1]) - output).sum() # some scalar function to optimize
loss.backward()
W1.grad
W2.grad
b1.grad
b2.grad
