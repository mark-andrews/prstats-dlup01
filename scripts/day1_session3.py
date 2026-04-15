import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import datasets, transforms

# load up data
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

# Optional: Plot images
image, label = train_data[0]
image.shape
plt.clf() # clear matplotlib image
plt.imshow(image[0,:,:], cmap='grey')
# look at some others
plt.imshow(train_data[10101][0][0,:,:], cmap='grey')
plt.imshow(train_data[37561][0][0,:,:], cmap='grey')

# Set up MLP
mlp = nn.Sequential(
    nn.Flatten(),
    # input to hidden layer weight
    nn.Linear(784, 128),
    # activation function at hidden layer
    nn.ReLU(),
    # hidden to output layer
    nn.Linear(128, 10)
    # softmax not necessary; but is implicit in this model
)

mlp2 = nn.Sequential(
    nn.Flatten(),
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

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters())

# for example
X, y = train_data[0] 
mlp(X)


# create data batch
from torch.utils.data import DataLoader

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

losses = []
max_epoch = 10
for epoch in range(max_epoch):
    epoch_loss = 0
    for X, y in train_loader:
        optimizer.zero_grad()
        loss = criterion(mlp(X), y) # Step 1: calculate loss function for batch
        loss.backward()             # Step 2: calculate gradient
        optimizer.step()            # Step 3: Gradient descent 
        epoch_loss += loss.item()
    losses.append(epoch_loss)
    print(epoch_loss)


plt.clf()
plt.plot(losses)

# how many parameters?
# [100352, 128, 1280, 10]
sum([p.numel() for p in mlp.parameters()])


correct = 0
for X, y in test_loader:
    preds = mlp(X).argmax(dim = 1)
    correct += (preds == y).sum().item()
accuracy = correct / len(test_data)
