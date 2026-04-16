import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


conv_layer = nn.Conv2d(1, 32, kernel_size=3, padding=1)

X = torch.randn(1, 1, 28, 28) # 1 greyscale 28x28 image
out = conv_layer(X)
out.shape

conv_layer2 = nn.Conv2d(1, 32, stride=2, kernel_size=3, padding=1)
out2 = conv_layer2(X)
out2.shape

pool = nn.MaxPool2d(kernel_size=2)
pool(out).shape

class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.bn1 = nn.BatchNorm2d(32) # batch norm function for conv layer 1
        self.bn2 = nn.BatchNorm2d(64) # batch norm function for conv layer 2
        self.fc = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.flatten(start_dim=1)
        return self.fc(x)


conv_net = CNN() 

# count the number of parameters
sum([p.numel() for p in conv_net.parameters() if p.requires_grad])

# Load data
# load up data
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

# Set up data loader

# create data batch

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# Initialize criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(conv_net.parameters())

# Loop through data
max_iter = 10
losses = []

for epoch in range(max_iter):
    epoch_loss = 0
    for X, y in train_loader:
        optimizer.zero_grad()
        loss = criterion(conv_net(X), y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    losses.append(epoch_loss)
    print(epoch_loss)

correct = 0
for X,y in test_loader:
    correct += (conv_net(X).argmax(dim=1) == y).sum()

accuracy = correct / len(test_data)
accuracy

# 32 filters on conv layer 1 after training
filters = conv_net.conv1.weight.data

# feature (filter) 1
plt.imshow(filters[0,0], cmap='grey', interpolation='nearest')
# feature (filter) 2
plt.imshow(filters[1,0], cmap='grey', interpolation='nearest')
