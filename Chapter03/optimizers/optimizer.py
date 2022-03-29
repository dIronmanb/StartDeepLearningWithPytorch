import torch.optim as optimizer


def get_optimizer(model_parameter, config):
    optimizer_name = config['optimizer']
    if optimizer_name == 'Adam':
        return optimizer.Adam(params = model_parameter,
                             lr = config['learning_rate'],
                             weight_decay = config['weight_decay']   
                            )
    elif optimizer_name == 'SGD':
        return optimizer.SGD(params = model_parameter,
                             lr = config['learning_rate'],
                             momentum=config['momentum'],
                             weight_decay=config['weight_decay']
                             )
    else:
        print("There is no name in Optimizers")
        
        