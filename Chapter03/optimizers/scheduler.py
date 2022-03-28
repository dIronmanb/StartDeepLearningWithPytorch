import torch.optim.lr_scheduler as scheduler



def get_scheduler(optimizer, config):
    scheduler_name = config['scheduler_name']
    
    if scheduler_name == 'LambdaLR':
        return scheduler.LambdaLR(optimizer = optimizer,
                                  lr_lambda = lambda epoch : 0.95 ** epoch
                                 )
    # elif scheduler_name  == '':
    #     return scheduler.MultiplicativeLR()
    elif scheduler_name == 'StepLR':
        return scheduler.StepLR(optimizer = optimizer,
                                step_size = 10,
                                gamma = 0.5)
    elif scheduler_name == 'MultiStepLR':
        return scheduler.MultiStepLR(optimizer=optimizer,
                                     milestones=[30,80],
                                     gamma = 0.5)
    elif scheduler_name == 'ExponentialLR':
        return scheduler.ExponentialLR(optimizer=optimizer,
                                       gamma = 0.5)
    elif scheduler_name == 'CosineAnnealingLR':
        return scheduler.CosineAnnealingLR(optimizer=optimizer,
                                           T_max = 50,
                                           eta_min = 0)
    