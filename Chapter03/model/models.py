'''
    1. CNN LAYER는 보통 conv -> relu , conv -> relu , ... 이다.
    2. 끝단에 maxpooling등을 넣어만들어서 다음과 같은 형태가 되면
    nn.Sequential(
        nn.conv
        nn.relu
        
        nn.conv
        nn.relu
    ) + maxpooling
    -->> 이를 BLOCK이라고 부른다.
    3. batch_size는 점점 크게 한다.
     (ex) 3 -> 8 -> 16 -> 32 -> 64 ...
     
    4. 같은 이름을 사용하여 객체가 서로 공유되지 아니하도록 조심하기
'''
import torch 
import torch.nn as nn
import torch.nn.functional as F

def get_cnn_model(config):
    name = config['model']
    if name == 'CNN_3':
        return CNN_3()
    elif name == 'CNN_5':
        return CNN_5()
    elif name == 'CNN_9':
        return CNN_9()
    elif name == 'CNN_12':
        return CNN_12()
    elif name == 'VGG11':
        vgg_types = config['VGG_types']
        return VGGnet(vgg_types[name], init_weights=True)
    else:
        print("There is no name in models")
   
class CNN_5(nn.Module):
    def __init__(self):
        super(CNN_5, self).__init__()
        # Output = (Input - Kernel_size + 2*Padding_size) / Stride + 1
        self.conv_in = nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = 3, padding = 1)
        self.conv_hidden_1 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, padding = 1)
        self.conv_hidden_2 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, padding = 1)
        self.conv_hidden_3 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, padding = 1)
        self.conv_out = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, padding = 1)
        
        # Output = (Input - kernel_size) / stride + 1
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(8 * 8 * 16, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
                
         
    def forward(self,x):      
        x = self.pool(F.relu(self.conv_in(x))) # 16 * 16 * 8
        
        x = F.relu(self.conv_hidden_1(x)) # 16 * 16 * 8
        x = F.relu(self.conv_hidden_2(x)) # 16 * 16 * 8
        x = F.relu(self.conv_hidden_3(x)) # 16 * 16 * 8
                
        x = self.pool(F.relu(self.conv_out(x))) # 8 * 8 * 16
        
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
        self.conv_hidden_1 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, padding = 1)
        self.conv_hidden_2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, padding = 1)
        self.conv_hidden_3 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = 1)
        self.conv_hidden_4 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1)
        self.conv_hidden_5 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1)
        self.conv_hidden_6 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
        self.conv_hidden_7 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)
        self.conv_out = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)
        
        # Output = (Input - kernel_size) / stride + 1
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(8 * 8 * 128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
                
         
        
    def forward(self,x):
        x = self.conv_in(x) # 32 * 32 * 8
        x = F.relu(x)
        x = self.pool(x) # 16 * 16 * 8
      
          
        x = F.relu(self.conv_hidden_1(x)) # 16 * 16 * 8    
        x = F.relu(self.conv_hidden_2(x)) # 16 * 16 * 8
        x = F.relu(self.conv_hidden_3(x)) # 16 * 16 * 8
        x = F.relu(self.conv_hidden_4(x)) # 16 * 16 * 8    
        x = F.relu(self.conv_hidden_5(x)) # 16 * 16 * 8
        x = F.relu(self.conv_hidden_6(x)) # 16 * 16 * 8
        x = F.relu(self.conv_hidden_7(x)) # 16 * 16 * 64
        
        x = self.conv_out(x) # 16 * 16 * 128
        x = F.relu(x)
        x = self.pool(x)  # 8 * 8 * 128
        
        
        
        x = x.view(-1, 8 * 8 * 128) # flatten
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
class VGGnet(nn.Module):
    def __init__(self, model, in_channels = 3, num_classes = 10, init_weights = True):
        super(VGGnet, self).__init__()
        self.in_channels = in_channels
        
        self.conv_layers = self.create_conv_layers(model)    
        
        self.fcs = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(4096, num_classes)
        )
        
        if init_weights:
            self._initialize_weights()
            
            
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 512)
        x = self.fcs(x)
        return x
    
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        
        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers += [nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                                     kernel_size = (3,3), stride = (1,1), padding = (1,1)),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))]
                
        return nn.Sequential(*layers)

if __name__ == '__main__':
    print('Quick Test...')
    input = torch.zeros([1,3,32,32], dtype = torch.float32)
    model = CNN_3()
    output = model(input)
    
    print('input_shape: {}, output_size: {}'
          .format(input.shape, output.shape))
