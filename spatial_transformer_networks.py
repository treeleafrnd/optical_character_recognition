
#Importing Dependencies
"""

import glob
import time
import sklearn.decomposition

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms

from PIL import Image
from skimage.util import random_noise

"""# Figure Setting"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
plt.ion()   # interactive mode

"""# To check whether the device is GPU or CPU"""

# inform the user if the notebook uses GPU or CPU.

def set_device():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  if device != "cuda":
    print("GPU is not enabled in this notebook. \n"
          "If you want to enable it, in the menu under `Runtime` -> \n"
          "`Hardware accelerator.` and select `GPU` from the dropdown menu")
  else:
    print("GPU is enabled in this notebook. \n"
          "If you want to disable it, in the menu under `Runtime` -> \n"
          "`Hardware accelerator.` and select `None` from the dropdown menu")

  return device

device = set_device()

"""#Mount the Drive"""

from google.colab import drive
drive.mount('/content/gdrive')

train_data = glob.glob("/content/gdrive/MyDrive/Spatial_Transformer_Networks/dataset/train/images/*.jpg")
test_data = glob.glob("/content/gdrive/MyDrive/Spatial_Transformer_Networks/dataset/test/images/*.jpg")

"""#Data Loading"""

# Training dataset
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=64,
                                           shuffle=True,
                                           num_workers=2)
# Test dataset
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=64,
                                          shuffle=True,
                                          num_workers=2)

"""#Spatial Transformer on Image
##Spatial Transformer on Network

"""

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.conv2_drop = nn.Dropout2d()
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)

    # Spatial transformer localization-network
    self.localization = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=7),
        nn.MaxPool2d(2, stride=2),
        nn.ReLU(True),
        nn.Conv2d(8, 10, kernel_size=5),
        nn.MaxPool2d(2, stride=2),
        nn.ReLU(True)
    )

    # Regressor for the 3 * 2 affine matrix
    self.fc_loc = nn.Sequential(
        nn.Linear(10 * 3 * 3, 32),
        nn.ReLU(True),
        nn.Linear(32, 3 * 2)
    )

    # Initialize the weights/bias with identity transformation
    self.fc_loc[2].weight.data.zero_()
    self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

  # Spatial transformer network forward function
  def stn(self, x):
    xs = self.localization(x)
    xs = xs.view(-1, 10 * 3 * 3)
    theta = self.fc_loc(xs)
    theta = theta.view(-1, 2, 3)

    grid = F.affine_grid(theta, x.size())
    x = F.grid_sample(x, grid)

    return x

  def forward(self, x):
    # transform the input
    x = self.stn(x)

    # Perform the usual forward pass
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    x = x.view(-1, 320)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)

"""#Training the model

Now, let’s use the SGD algorithm to train the model. The network is learning the classification task in a supervised way. In the same time the model is learning STN automatically in an end-to-end fashion.
"""

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)

"""#Train and Test functions for STN

###Train Function
"""

def train(train_loader, optimizer, epoch, device):

  model.train()
  for batch_idx, (*data, target) in enumerate(train_loader):
    target = np.asarray(target)
    target = torch.from_numpy(target.astype('long'))    
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % 500 == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item()))

"""### Test Function"""

def test(test_loader, device):

  with torch.no_grad():
    model.eval()
    test_loss, correct = 0, 0

    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)

      # sum up batch loss
      test_loss += F.nll_loss(output, target, size_average=False).item()
      # get the index of the max log-probability
      pred = output.max(1, keepdim=True)[1]
      correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          .format(test_loss, correct, len(test_loader.dataset),
                  100. * correct / len(test_loader.dataset)))

"""#Run Train and Test the data"""

num_epochs = 20
for epoch in range(1, num_epochs + 1):
  train(train_loader, optimizer, epoch, device)
  test(test_loader, device)
