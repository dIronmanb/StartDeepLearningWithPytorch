import torch.nn as nn

def get_loss(config, params = None):
    name = config['loss']
    if name == 'MSELoss':
        return nn.MSELoss()
    elif name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    elif name == 'Softmax':
        return nn.Softmax()
    else:
        print("There is no name in loss")   
        
