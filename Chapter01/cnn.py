import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Get Dataset from CIFAR10

def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding = "bytes")
    return dict

def pickle_to_images_and_labels(root):
    # Get dataset
    data = unpickle(root)
    # normalize
    data_images = data[b'data'] / 255 
    # reshape data >> -1 * 3 * 32 * 32
    # num of data, colour, width, height
    data_images = data_images.reshape(-1, 3, 32, 32).astype("float32")
    data_labels = data[b'labels']
    return data_images, data_labels

train_images_and_labels = []
for i in range(5):
    train_images_and_labels.append(pickle_to_images_and_labels("/data/CNN_model/dataset/cifar-10-batches-py/data_batch_" + str(i + 1)))

train_images = np.concatenate([i[0] for i in train_images_and_labels], axis = 0)
train_labels = np.concatenate([i[1] for i in train_images_and_labels], axis = 0)
print(train_images.shape)
print(train_labels.shape)

test_images, test_labels = pickle_to_images_and_labels("/data/CNN_model/dataset/cifar-10-batches-py/test_batch")
test_images = np.concatenate([test_images], axis = 0)
test_labels = np.concatenate([test_labels], axis = 0)

# Convert numpy to tensor
train_images_tensor = torch.tensor(train_images)
train_labels_tensor = torch.tensor(train_labels)
train_tensor = TensorDataset(train_images_tensor, train_labels_tensor)
train_loader = DataLoader(train_tensor, batch_size = 64, num_workers=0, shuffle = True)

test_images_tensor = torch.tensor(test_images)


# CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 12, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 12, out_channels = 20, kernel_size = 3, padding = 1)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.fc1 = nn.Linear(8 * 8 * 20, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32,10)
        
         
        
    def forward(self,x):
        x = F.relu(self.conv1(x)) # 32 * 32 * 12
        x = self.pool(x)          # 16 * 16 * 12
        x = F.relu(self.conv2(x)) # 16 * 16 * 20
        x = self.pool(x)          # 8 * 8 * 20
        
        x = x.view(-1, 8 * 8 * 20)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x)
        return x
         
         

def train(model, train_loader, optimizer):
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE, dtype = torch.int64)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                   epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()
            ))
            
    loss_list.append(loss.item())
            
def test(model, test_images_tensor):
    model.eval()
    result = []
    
    with torch.no_grad():
        for data in test_images_tensor:
            data = data.to(DEVICE)
            output = model(data.view(-1, 3, 32, 32))
            prediction = output.max(1, keepdim = True)[1]
            result.append(prediction.tolist())
    return result


# Train and Test
DEVICE = torch.device("cuda" if torch.cuda.is_available else "cpu")
model = CNN().to(DEVICE)
EPOCHS = 100
optimizer = optim.Adam(model.parameters(), lr = 0.001)

loss_list = []
accuracy_list = []

for epoch in range(1, EPOCHS + 1):
    train(model, train_loader, optimizer)
    
    predicted = test(model, test_images_tensor)
    accuracy = accuracy_score(test_labels, np.squeeze(predicted))
    accuracy_list.append(accuracy * 100)
    # print("Accuracy: {:.2f}%".format(accuracy * 100))
    print()

# test_predict_result = test(model, test_images_tensor)
# accuracy = accuracy_score(test_labels, np.squeeze(test_predict_result))
# print(accuracy)
loss_list = np.round(np.array(loss_list), 3)
print(loss_list)
print(accuracy_list)

f, axes = plt.subplots(1,2, figsize = (20,10))

axes[0].plot(range(1,EPOCHS+1), loss_list, color = 'red', label = 'train_loss')
axes[0].axis([-5, 105, -1, 4])
axes[0].set_xlabel('epoch')
axes[0].set_ylabel('loss')
axes[0].legend()
axes[1].plot(range(1,EPOCHS+1), accuracy_list, color = 'blue', label = "accuracy")
axes[1].axis([-5, 105, -5, 105])
axes[1].set_xlabel('epoch')
axes[1].set_ylabel('accuracy')
axes[1].legend()

plt.show()
plt.savefig('CIFAR10_result_0.png')