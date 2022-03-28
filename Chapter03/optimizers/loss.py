import torch.nn as nn

def get_loss_function(name, params = None):
    if name == 'MSELoss':
        return nn.MSELoss()
    elif name =='CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    else:
        print("There is no name in loss_functions")
