import torch.nn as nn
import torch.nn.functional as F

def get_cnn_model(name):
    if name == 'CNN_3':
        return CNN_3()
    elif name == 'CNN_5':
        return CNN_5()
    elif name == 'CNN_9':
        return CNN_9()
    elif name == 'CNN_12':
        return CNN_12()
    else:
        print("There is no name in models")

# class CNN_3(nn.Module):
#     def __init__(self):
#         super(CNN_3, self).__init__()
#         # Output = (Input - Kernel_size + 2*Padding_size) / Stride + 1
#         self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = 3, padding = 1)
#         self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, padding = 1)
#         self.conv3 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, padding = 1)
        
#          # Output = (Input - kernel_size) / stride + 1
#         self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
#         self.fc1 = nn.Linear(4 * 4 * 8, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, 10)
                
         
        
#     def forward(self,x):
#         x = self.conv1(x) # 32 * 32 * 8
#         x = self.pool(x) # 16 * 16 * 8
#         x = F.relu(x)
        
#         x = self.conv2(x) # 16 * 16 * 8
#         x = self.pool(x) # 8 * 8 * 8
#         x = F.relu(x)
        
#         x = self.conv2(x) # 8 * 8 * 8
#         x = self.pool(x)  # 4 * 4 * 8
#         x = F.relu(x)
        
        
#         x = x.view(-1, 4 * 4 * 8) # flatten
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         x = F.relu(x)
#         x = self.fc3(x)
#         x = F.log_softmax(x)
#         return x
    
class CNN_5(nn.Module):
    def __init__(self):
        super(CNN_5, self).__init__()
        # Output = (Input - Kernel_size + 2*Padding_size) / Stride + 1
        self.conv_in = nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = 3, padding = 1)
        self.conv_hidden = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, padding = 1)
        self.conv_out = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, padding = 1)
        
        # Output = (Input - kernel_size) / stride + 1
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(8 * 8 * 16, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
                
         
        
    def forward(self,x):
        x = self.conv_in(x) # 32 * 32 * 8
        x = self.pool(x) # 16 * 16 * 8
        x = F.relu(x)
        
        x = F.relu(self.conv_hidden(x)) # 16 * 16 * 8
        x = F.relu(self.conv_hidden(x)) # 16 * 16 * 8
        x = F.relu(self.conv_hidden(x)) # 16 * 16 * 8
        
        x = self.conv_out(x) # 16 * 16 * 16
        x = self.pool(x)  # 8 * 8 * 16
        x = F.relu(x)
        
        
        x = x.view(-1, 8 * 8 * 16) # flatten
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x)
        return x
    
class CNN_9(nn.Module):
    def __init__(self):
        super(CNN_9, self).__init__()
        # Output = (Input - Kernel_size + 2*Padding_size) / Stride + 1
        self.conv_in = nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = 3, padding = 1)
        self.conv_hidden = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, padding = 1)
        self.conv_out = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, padding = 1)
        
        # Output = (Input - kernel_size) / stride + 1
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(8 * 8 * 16, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
                
         
        
    def forward(self,x):
        x = self.conv_in(x) # 32 * 32 * 8
        x = self.pool(x) # 16 * 16 * 8
        x = F.relu(x)
        
        for _ in range(7):
            x = F.relu(self.conv_hidden(x)) # 16 * 16 * 8
        
        # x = F.relu(self.conv_hidden(x)) # 16 * 16 * 8    
        # x = F.relu(self.conv_hidden(x)) # 16 * 16 * 8
        # x = F.relu(self.conv_hidden(x)) # 16 * 16 * 8
        # x = F.relu(self.conv_hidden(x)) # 16 * 16 * 8    
        # x = F.relu(self.conv_hidden(x)) # 16 * 16 * 8
        # x = F.relu(self.conv_hidden(x)) # 16 * 16 * 8
        # x = F.relu(self.conv_hidden(x)) # 16 * 16 * 8
        
        x = self.conv_out(x) # 16 * 16 * 16
        x = self.pool(x)  # 8 * 8 * 16
        x = F.relu(x)
        
        
        x = x.view(-1, 8 * 8 * 16) # flatten
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x)
        return x
    
class CNN_12(nn.Module):
    
    def __init__(self):
        super(CNN_12, self).__init__()
        # Output = (Input - Kernel_size + 2*Padding_size) / Stride + 1
        self.conv_in = nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = 3, padding = 1)
        self.conv_hidden = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, padding = 1)
        self.conv_out = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, padding = 1)
        
        # Output = (Input - kernel_size) / stride + 1
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(8 * 8 * 16, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
                
         
        
    def forward(self,x):
        x = self.conv_in(x) # 32 * 32 * 8
        x = self.pool(x) # 16 * 16 * 8
        x = F.relu(x)
        
        for _ in range(10):
            x = F.relu(self.conv_hidden(x)) # 16 * 16 * 8
        
        # x = F.relu(self.conv_hidden(x)) # 16 * 16 * 8    
        
        x = self.conv_out(x) # 16 * 16 * 16
        x = self.pool(x)  # 8 * 8 * 16
        x = F.relu(x)
        
        
        x = x.view(-1, 8 * 8 * 16) # flatten
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x)
        return x
       
    
class CNN_3(nn.Module):
    def __init__(self): 
        super(CNN_3, self).__init__()
        
        # # Output = (Input - Kernel_size + 2*Padding_size) / Stride + 1
        # self.conv_in = nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = 3, padding = 1)
        #input = 3, output = 6, kernal = 5
        self.conv1 = nn.Conv2d(3, 6, 5) # 32 * 32 * 6
        
        
        # # Output = (Input - kernel_size) / stride + 1
        # self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)        
        #kernal = 2, stride = 2, padding = 0 (default)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #input feature, output feature
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        
    # 값 계산
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x))) 
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x)) 
        x = self.fc3(x) 
        return x


