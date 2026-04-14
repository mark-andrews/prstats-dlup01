import torch
import torchvision
from torchvision import datasets, transforms

transform = transforms.ToTensor()
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

image, label = train_data[0]
image.shape
plt.clf() # clear matplotlib image
plt.imshow(image[0,:,:], cmap='grey')
# look at some others
plt.imshow(train_data[10101][0][0,:,:], cmap='grey')
plt.imshow(train_data[37561][0][0,:,:], cmap='grey')

# Set up MLP
import torch.nn as nn

mlp = nn.Sequential(
    # input to hidden layer weight
    nn.Linear(784, 128),
    # activation function at hidden layer
    nn.ReLU(),
    # hidden to output layer
    nn.Linear(128, 10)
    # softmax not necessary; but is implicit in this model
)

mlp2 = nn.Sequential(
    # input to hidden layer weight
    nn.Linear(784, 128),
    # activation function at hidden layer
    nn.ReLU(),
    # hidden_1 to hidden_2
    nn.Linear(128, 64),
    nn.ReLU(),
    # hidden_2 to hidden_3
    nn.Linear(64, 128),
    nn.ReLU(),
    # hidden_3 to output layer
    nn.Linear(128, 10)
    # softmax not necessary; but is implicit in this model
)