import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml 

def get_VGG(config):
    name = config['model']
    model_list = config['VGG_types']
    
    if name == 'VGG11':
        return VGGnet(model_list[name])
    elif name == 'VGG13':
        return VGGnet(model_list[name])
    elif name == 'VGG16':
        return VGGnet(model_list[name])
    elif name == 'VGG19':
        return VGGnet(model_list[name])
    else:
        print("There is no name in models")
        
        
 
class VGGnet(nn.Module):
    def __init__(self, model, in_channels = 3, num_classes = 36, init_weights = True):
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
                
                
# Open config file -> quick test
def open_config_file():
    with open("/data/Github_Management/StartDeepLearningWithPytorch/Chapter04/config/config.yaml", 'r', encoding = 'utf-8') as stream:
        try:
            config = yaml.safe_load(stream) # return into Dict
        except yaml.YAMLError as exc:
            print(exc)  
    return config['VGG_types']  

    
    
if __name__ == '__main__':
    print('Quick Test...')
    
    models = open_config_file()
    model = VGGnet(models['VGG19'])
    print(model)
    
    input = torch.zeros([1,3,32,32], dtype = torch.float32)
    # model = VGG_19(32, 3)
    output = model(input)
    
    print('input_shape: {}, output_size: {}'
          .format(input.shape, output.shape))
    