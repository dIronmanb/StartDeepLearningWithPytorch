import torch.optim.lr_scheduler as scheduler

def get_scheduler(optimizer, config):
    name = config['scheduler']
    
    if name == 'LamdbaLR':
        return scheduler.LambdaLR(optimizer = optimizer,
                                  lr_lambda = lambda epoch : 0.95 ** epoch )
    elif name == 'StepLR':
        return scheduler.StepLR(optimizer = optimizer,
                                step_size = 10,
                                gamma = 0.5)
    elif name == 'MultiStepLR':
        return scheduler.MultiStepLR(optimizer = optimizer,
                                     milestones = [30, 80],
                                     gamma = 0.5)
    elif name == 'ExponentialLR':
        return scheduler.ExponentialLR(optimizer = optimizer,
                                       gamma = 0.5)
    elif name == 'CosineAnnealingLR':
        return scheduler.CosineAnnealingWarmRestarts(optimizer = optimizer,
                                                     T_max = 50,
                                                     eta_min = 0)
    else:
        print("There was no name in scheduler")
    
    
    
    