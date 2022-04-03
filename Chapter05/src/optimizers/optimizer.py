import torch.optim as optimizer


def get_optimizer(model_paramter, config):
    name = config['optimizer']
    
    if name == 'Adam':
        return optimizer.Adam(params = model_paramter,
                              lr = config['learning_rate'],
                              weight_decay = config['weight_decay'])
        
    elif name == 'SGD':
        return optimizer.SGD(params = model_paramter,
                             lr = config['learning_rate'],
                             momentum = config['momentum'],
                             weight_decay = config['weight_decay'])
        
    else:
        print("There is no name in optimizer")